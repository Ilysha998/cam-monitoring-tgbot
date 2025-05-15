import cv2
import requests
import numpy as np
import asyncio
import json
import os
import time
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
from requests.auth import HTTPBasicAuth
from urllib.parse import urlparse
from telegram.error import TelegramError, TimedOut

CONFIG_FILE = "cam_config.json"

def load_config():
    default_config = {
        "token": "YOUR_BOT_TOKEN",
        "admin_chat_id": "YOUR_CHAT_ID",
        "cameras": {},
        "motion_enabled": True,
        "cooldown": 30,
        "threshold": 30000,
        "check_interval": 5,
        "timeout": 10,
        "draw_contours": True,
        "contour_color": [0, 255, 0],
        "contour_thickness": 2,
        "min_contour_area": 1000,
        "frame_width": 800,
        "frame_height": 600
    }
    
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return {**default_config, **json.load(f)}
    except Exception as e:
        print(f"Ошибка загрузки конфига: {e}")
    
    return default_config

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Ошибка сохранения конфига: {e}")

class CameraSystem:
    def __init__(self):
        self.config = load_config()
        self.bot = None
        self.prev_frames = {}
        self.camera_status = {}
        self.last_notification = 0
        self.mjpeg_streams = {}
        self.lock = asyncio.Lock()
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.motion_history = {}

    async def init_bot(self):
        try:
            self.bot = Bot(token=self.config["token"])
            await self.bot.get_me()
            print("Бот инициализирован")
        except Exception as e:
            print(f"Ошибка инициализации бота: {e}")
            raise

    async def check_cameras_status(self):
        async with self.lock:
            for cam_id, cam_data in self.config["cameras"].items():
                try:
                    auth = HTTPBasicAuth(cam_data.get("user", ""), cam_data.get("password", ""))
                    response = requests.head(
                        cam_data["url"],
                        timeout=self.config["timeout"],
                        auth=auth
                    )
                    self.camera_status[cam_id] = (response.status_code == 200)
                except Exception as e:
                    self.camera_status[cam_id] = False

    async def process_camera(self, cam_id, cam_data):
        try:
            if self.is_mjpeg_url(cam_data["url"]):
                return await self.process_mjpeg_camera(cam_id, cam_data)
            else:
                return await self.process_jpeg_camera(cam_data)
        except Exception as e:
            print(f"Ошибка обработки камеры {cam_id}: {e}")
            return None

    async def process_jpeg_camera(self, cam_data):
        try:
            auth = HTTPBasicAuth(cam_data.get("user", ""), cam_data.get("password", ""))
            response = requests.get(
                cam_data["url"],
                timeout=self.config["timeout"],
                auth=auth
            )
            if response.status_code == 200:
                return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Ошибка JPEG камеры: {e}")
            return None

    async def process_mjpeg_camera(self, cam_id, cam_data):
        async with self.lock:
            if cam_id in self.mjpeg_streams and self.mjpeg_streams[cam_id].isOpened():
                stream = self.mjpeg_streams[cam_id]
                ret, frame = stream.read()
                if ret:
                    return frame
                else:
                    stream.release()
                    del self.mjpeg_streams[cam_id]

            try:
                stream = cv2.VideoCapture(cam_data["url"])
                if stream.isOpened():
                    ret, frame = stream.read()
                    if ret:
                        self.mjpeg_streams[cam_id] = stream
                        return frame
            except Exception as e:
                print(f"Ошибка MJPEG потока {cam_id}: {e}")
            return None

    def is_mjpeg_url(self, url):
        parsed = urlparse(url)
        return 'mjpg' in parsed.path.lower() or 'mjpeg' in parsed.path.lower()

    async def check_motion(self):
        while True:
            await self.check_cameras_status()
            
            if self.config["motion_enabled"]:
                for cam_id, cam_data in self.config["cameras"].items():
                    if self.camera_status.get(cam_id, False):
                        frame = await self.process_camera(cam_id, cam_data)
                        if frame is not None:
                            motion_detected, processed_frame = self.calculate_motion(frame, cam_id)
                            if motion_detected:
                                await self.send_alert(cam_id, processed_frame)
            
            await asyncio.sleep(self.config["check_interval"])

    def calculate_motion(self, frame, cam_id):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if cam_id not in self.prev_frames:
                self.prev_frames[cam_id] = gray
                return False, frame

            prev_gray = self.prev_frames[cam_id]
            frame_diff = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]

            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            thresh = cv2.erode(thresh, kernel, iterations=1)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = False
            processed_frame = frame.copy()

            for contour in contours:
                if cv2.contourArea(contour) < self.config["min_contour_area"]:
                    continue
                motion_detected = True
                
                if self.config["draw_contours"]:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(processed_frame, (x, y), (x+w, y+h),
                                self.config["contour_color"],
                                self.config["contour_thickness"])

            self.prev_frames[cam_id] = gray
            return motion_detected, processed_frame

        except Exception as e:
            print(f"Ошибка детекции движения: {e}")
            return False, frame

    async def send_alert(self, cam_id, frame):
        current_time = time.time()
        if current_time - self.last_notification < self.config["cooldown"]:
            return

        try:
            _, buffer = cv2.imencode('.jpg', frame)
            await self.bot.send_photo(
                chat_id=self.config["admin_chat_id"],
                photo=bytes(buffer),
                caption=f"⚠️ Движение на камере {cam_id} ({time.strftime('%Y-%m-%d %H:%M:%S')})"
            )
            self.last_notification = current_time
        except Exception as e:
            print(f"Ошибка отправки уведомления: {e}")

    async def cleanup(self):
        async with self.lock:
            for cam_id, stream in self.mjpeg_streams.items():
                if stream.isOpened():
                    stream.release()
            self.mjpeg_streams.clear()

    async def get_snapshot(self, cam_id):
        if cam_id not in self.config["cameras"]:
            return None
        return await self.process_camera(cam_id, self.config["cameras"][cam_id])

    async def toggle_motion_detection(self, state):
        self.config["motion_enabled"] = state
        save_config(self.config)
        return True

    async def main_menu(self, update: Update, message_id=None):
        motion_state = "ВКЛ" if self.config["motion_enabled"] else "ВЫКЛ"
        buttons = [
            [InlineKeyboardButton("📷 Список камер", callback_data='list_cams'),
             InlineKeyboardButton("📊 Статус системы", callback_data='status')],
            [InlineKeyboardButton(f"🎥 Детекция: {motion_state}", callback_data='toggle_motion'),
             InlineKeyboardButton("📸 Сделать снимок", callback_data='snapshot_menu')],
            [InlineKeyboardButton("⚙️ Настройки", callback_data='settings'),
             InlineKeyboardButton("🔄 Обновить", callback_data='refresh')]
        ]
        keyboard = InlineKeyboardMarkup(buttons)
        
        text = "📹 Главное меню управления камерами"
        if message_id:
            await self.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=message_id,
                text=text,
                reply_markup=keyboard
            )
        else:
            await update.message.reply_text(text, reply_markup=keyboard)

system = CameraSystem()

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = update.effective_chat.id

    if str(chat_id) != system.config["admin_chat_id"]:
        return

    try:
        if data == 'list_cams':
            await list_cameras(update, context)
        elif data == 'status':
            await status(update, context)
        elif data == 'toggle_motion':
            new_state = not system.config["motion_enabled"]
            await system.toggle_motion_detection(new_state)
            await system.main_menu(update, query.message.message_id)
        elif data == 'snapshot_menu':
            await snapshot_menu(update, context)
        elif data == 'settings':
            await settings(update, context)
        elif data == 'refresh':
            await system.check_cameras_status()
            await system.main_menu(update, query.message.message_id)
        elif data.startswith('snapshot_'):
            cam_id = data.split('_')[1]
            await send_snapshot(update, context, cam_id)
        elif data == 'back':
            await system.main_menu(update, query.message.message_id)
    except Exception as e:
        print(f"Ошибка обработки callback: {e}")
        await query.edit_message_text("⚠️ Произошла ошибка, попробуйте позже")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != system.config["admin_chat_id"]:
        return
    await system.main_menu(update)

async def list_cameras(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not system.config["cameras"]:
        await update.message.reply_text("Нет добавленных камер")
        return

    message = ["Список камер:"]
    for name, data in system.config["cameras"].items():
        status = "🟢 Онлайн" if system.camera_status.get(name, False) else "🔴 Оффлайн"
        auth = " (требуется auth)" if data.get("user") else ""
        message.append(f"{status} {name}: {data['url']}{auth}")

    await update.message.reply_text("\n".join(message))

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    online_cams = sum(system.camera_status.values())
    total_cams = len(system.config["cameras"])

    status_text = [
        f"🔹 Детекция движения: {'ВКЛ' if system.config['motion_enabled'] else 'ВЫКЛ'}",
        f"🔹 Камер: {total_cams} ({online_cams} онлайн)",
        f"🔹 Порог чувствительности: {system.config['threshold']}",
        f"🔹 Интервал проверки: {system.config['check_interval']} сек",
        f"🔹 Таймаут подключения: {system.config['timeout']} сек",
        f"🔹 Задержка между уведомлениями: {system.config['cooldown']} сек",
    ]

    await update.message.reply_text("Статус системы:\n\n" + "\n".join(status_text))

async def snapshot_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    buttons = []
    for cam_id in system.config["cameras"]:
        status_icon = "🟢" if system.camera_status.get(cam_id, False) else "🔴"
        buttons.append([InlineKeyboardButton(f"{status_icon} {cam_id}", callback_data=f'snapshot_{cam_id}')])
    buttons.append([InlineKeyboardButton("🔙 Назад", callback_data='back')])
    
    keyboard = InlineKeyboardMarkup(buttons)
    await context.bot.edit_message_text(
        chat_id=update.effective_chat.id,
        message_id=update.callback_query.message.message_id,
        text="Выберите камеру для снимка:",
        reply_markup=keyboard
    )

async def send_snapshot(update: Update, context: ContextTypes.DEFAULT_TYPE, cam_id: str):
    query = update.callback_query
    await query.answer("Делаем снимок...")
    
    frame = await system.get_snapshot(cam_id)
    if frame is None:
        await query.edit_message_text("❌ Не удалось получить снимок")
        return

    try:
        _, buffer = cv2.imencode('.jpg', frame)
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=bytes(buffer),
            caption=f"📸 Снимок с камеры {cam_id}"
        )
    except Exception as e:
        print(f"Ошибка отправки фото: {e}")
        await query.edit_message_text("❌ Ошибка отправки снимка")

async def main():
    await system.init_bot()
    
    app = Application.builder().token(system.config["token"]).build()

    # Регистрация обработчиков
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(CommandHandler("list", list_cameras))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("snapshot", snapshot_menu))

    # Запуск фоновых задач
    asyncio.create_task(system.check_motion())

    try:
        # Инициализация приложения
        await app.initialize()
        print("Бот запущен!")
        
        # Запуск опроса обновлений
        await app.start()
        await app.updater.start_polling()
        
        # Бесконечный цикл ожидания
        while True:
            await asyncio.sleep(3600)  # 1 час
            
    except (KeyboardInterrupt, SystemExit):
        # Корректное завершение
        await app.updater.stop()
        await app.stop()
    finally:
        # Финализация приложения
        await app.shutdown()
        await system.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен")