import cv2
import requests
import numpy as np
import asyncio
import json
import os
import time
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from requests.auth import HTTPBasicAuth

# Конфигурационный файл
CONFIG_FILE = "cam_config.json"

def load_config():
    """Загрузка конфигурации"""
    default_config = {
        "token": "YOUR_BOT_TOKEN",
        "admin_chat_id": "YOUR_CHAT_ID",
        "cameras": {},
        "motion_enabled": True,
        "cooldown": 30,
        "threshold": 30000,
        "check_interval": 5,
        "timeout": 10
    }
    
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return {**default_config, **json.load(f)}
    except Exception as e:
        print(f"Ошибка загрузки конфига: {e}")
    
    return default_config

def save_config(config):
    """Сохранение конфигурации"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Ошибка сохранения конфига: {e}")

class CameraSystem:
    def __init__(self):
        self.config = load_config()
        self.bot = None
        self.prev_frames = {}
        self.camera_status = {}
        self.last_notification = 0

    async def init_bot(self):
        """Инициализация бота"""
        try:
            self.bot = Bot(token=self.config["token"])
            await self.bot.get_me()  # Проверка подключения
        except Exception as e:
            print(f"Ошибка инициализации бота: {e}")
            raise

    async def check_cameras_status(self):
        """Проверка состояния всех камер"""
        for cam_id, cam_data in self.config["cameras"].items():
            try:
                response = requests.head(
                    cam_data["url"],
                    timeout=self.config["timeout"],
                    auth=HTTPBasicAuth(cam_data.get("user", ""), cam_data.get("password", ""))
                )
                self.camera_status[cam_id] = (response.status_code == 200)
            except:
                self.camera_status[cam_id] = False

    async def process_camera(self, cam_id, cam_data):
        """Обработка одной камеры"""
        try:
            response = requests.get(
                cam_data["url"],
                timeout=self.config["timeout"],
                auth=HTTPBasicAuth(cam_data.get("user", ""), cam_data.get("password", ""))
            )
            if response.status_code != 200:
                return None

            frame = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            return frame
        except:
            return None

    async def check_motion(self):
        """Основной цикл проверки движения"""
        while True:
            await self.check_cameras_status()
            
            if self.config["motion_enabled"]:
                for cam_id, cam_data in self.config["cameras"].items():
                    if self.camera_status.get(cam_id, False):
                        frame = await self.process_camera(cam_id, cam_data)
                        if frame is not None:
                            await self.detect_and_alert(cam_id, frame)
            
            await asyncio.sleep(self.config["check_interval"])

    async def detect_and_alert(self, cam_id, current_frame):
        """Детекция движения и отправка уведомления"""
        if cam_id in self.prev_frames:
            motion = self.calculate_motion(self.prev_frames[cam_id], current_frame)
            if motion > self.config["threshold"]:
                await self.send_alert(cam_id, current_frame)

        self.prev_frames[cam_id] = current_frame

    def calculate_motion(self, prev_frame, current_frame):
        """Вычисление уровня движения"""
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_prev, gray_current)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        return np.sum(thresh)

    async def send_alert(self, cam_id, frame):
        """Отправка уведомления"""
        current_time = time.time()
        if current_time - self.last_notification < self.config["cooldown"]:
            return

        alert_file = f"alert_{cam_id}.jpg"
        cv2.imwrite(alert_file, frame)
        
        try:
            with open(alert_file, 'rb') as photo:
                await self.bot.send_photo(
                    chat_id=self.config["admin_chat_id"],
                    photo=photo,
                    caption=f"⚠️ Движение на камере {cam_id}"
                )
            self.last_notification = current_time
        except Exception as e:
            print(f"Ошибка отправки: {e}")
        finally:
            if os.path.exists(alert_file):
                os.remove(alert_file)

    async def get_snapshot(self, cam_id):
        """Получение снимка с камеры"""
        if cam_id not in self.config["cameras"]:
            return None
        return await self.process_camera(cam_id, self.config["cameras"][cam_id])

    async def set_sensitivity(self, threshold):
        """Установка порога чувствительности"""
        try:
            threshold = int(threshold)
            if 1000 <= threshold <= 100000:
                self.config["threshold"] = threshold
                save_config(self.config)
                return True
            return False
        except ValueError:
            return False

    async def toggle_motion_detection(self, state):
        """Переключение детекции движения"""
        self.config["motion_enabled"] = state
        save_config(self.config)
        return True

# Обработчики команд Telegram
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    if str(update.effective_chat.id) != system.config["admin_chat_id"]:
        return

    commands = [
        "/start - Показать это сообщение",
        "/add <name> <url> [user] [password] - Добавить камеру",
        "/remove <name> - Удалить камеру",
        "/list - Список камер с статусом",
        "/status - Статус системы",
        "/motion on|off - Включить/выключить детекцию",
        "/sensitivity <value> - Установить порог (1000-100000)",
        "/snapshot <name> - Получить текущий кадр",
        "/check - Проверить состояние камер"
    ]
    
    await update.message.reply_text("📹 Система мониторинга камер\n\n" + "\n".join(commands))

async def list_cameras(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /list"""
    if str(update.effective_chat.id) != system.config["admin_chat_id"]:
        return

    if not system.config["cameras"]:
        await update.message.reply_text("Нет добавленных камер")
        return

    message = ["Список камер:"]
    for name, data in system.config["cameras"].items():
        status = "🟢" if system.camera_status.get(name, False) else "🔴"
        auth = " (требуется auth)" if data.get("user") else ""
        message.append(f"{status} {name}: {data['url']}{auth}")

    await update.message.reply_text("\n".join(message))

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /status"""
    if str(update.effective_chat.id) != system.config["admin_chat_id"]:
        return

    online_cams = sum(system.camera_status.values())
    total_cams = len(system.config["cameras"])

    status_text = [
        f"🔹 Детекция движения: {'ВКЛ' if system.config['motion_enabled'] else 'ВЫКЛ'}",
        f"🔹 Камер: {total_cams} ({online_cams} онлайн)",
        f"🔹 Порог чувствительности: {system.config['threshold']}",
        f"🔹 Интервал проверки: {system.config['check_interval']} сек",
        f"🔹 Таймаут подключения: {system.config['timeout']} сек",
        f"🔹 Задержка между уведомлениями: {system.config['cooldown']} сек"
    ]

    await update.message.reply_text("Статус системы:\n\n" + "\n".join(status_text))

async def snapshot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /snapshot"""
    if str(update.effective_chat.id) != system.config["admin_chat_id"]:
        return

    if not context.args:
        await update.message.reply_text("Укажите имя камеры: /snapshot <name>")
        return

    cam_id = context.args[0]
    if cam_id not in system.config["cameras"]:
        await update.message.reply_text(f"Камера {cam_id} не найдена")
        return

    if not system.camera_status.get(cam_id, False):
        await update.message.reply_text(f"Камера {cam_id} в данный момент оффлайн")
        return

    frame = await system.get_snapshot(cam_id)
    if frame is None:
        await update.message.reply_text("Не удалось получить кадр с камеры")
        return

    snapshot_file = f"snapshot_{cam_id}.jpg"
    cv2.imwrite(snapshot_file, frame)
    
    try:
        with open(snapshot_file, 'rb') as photo:
            await update.message.reply_photo(photo, caption=f"Кадр с камеры {cam_id}")
    finally:
        if os.path.exists(snapshot_file):
            os.remove(snapshot_file)

async def set_sensitivity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /sensitivity"""
    if str(update.effective_chat.id) != system.config["admin_chat_id"]:
        return

    if not context.args:
        current = system.config["threshold"]
        await update.message.reply_text(
            f"Текущая чувствительность: {current}\n"
            "Используйте: /sensitivity <значение от 1000 до 100000>"
        )
        return

    try:
        new_value = int(context.args[0])
        if await system.set_sensitivity(new_value):
            await update.message.reply_text(f"Порог чувствительности изменён на {new_value}")
        else:
            await update.message.reply_text("Недопустимое значение! Допустимый диапазон: 1000-100000")
    except ValueError:
        await update.message.reply_text("Введите числовое значение!")

async def toggle_motion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /motion"""
    if str(update.effective_chat.id) != system.config["admin_chat_id"]:
        return

    if not context.args:
        await update.message.reply_text("Используйте: /motion on|off")
        return

    state = context.args[0].lower()
    if state in ["on", "off"]:
        await system.toggle_motion_detection(state == "on")
        await update.message.reply_text(f"Детекция движения {'включена' if state == 'on' else 'выключена'}")
    else:
        await update.message.reply_text("Используйте: /motion on|off")

async def check_cameras(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /check"""
    if str(update.effective_chat.id) != system.config["admin_chat_id"]:
        return

    await update.message.reply_text("Проверяем состояние камер...")
    await system.check_cameras_status()
    await list_cameras(update, context)

async def add_camera(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /add"""
    if str(update.effective_chat.id) != system.config["admin_chat_id"]:
        return

    if len(context.args) < 2:
        await update.message.reply_text("Использование: /add <имя> <url> [user] [password]")
        return

    cam_name = context.args[0]
    cam_url = context.args[1]
    user = context.args[2] if len(context.args) > 2 else ""
    password = context.args[3] if len(context.args) > 3 else ""

    system.config["cameras"][cam_name] = {
        "url": cam_url,
        "user": user,
        "password": password
    }
    save_config(system.config)

    # Проверяем новую камеру
    is_online = await system.check_camera(cam_name, system.config["cameras"][cam_name])
    system.camera_status[cam_name] = is_online
    
    status = "🟢" if is_online else "🔴"
    await update.message.reply_text(f"Камера {cam_name} добавлена! Статус: {status}")

async def remove_camera(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /remove"""
    if str(update.effective_chat.id) != system.config["admin_chat_id"]:
        return

    if not context.args:
        await update.message.reply_text("Укажите имя камеры: /remove <name>")
        return

    cam_id = context.args[0]
    if cam_id in system.config["cameras"]:
        del system.config["cameras"][cam_id]
        save_config(system.config)
        if cam_id in system.camera_status:
            del system.camera_status[cam_id]
        await update.message.reply_text(f"Камера {cam_id} удалена")
    else:
        await update.message.reply_text(f"Камера {cam_id} не найдена")

async def main():
    """Основная функция"""
    global system
    system = CameraSystem()
    
    try:
        await system.init_bot()
    except Exception as e:
        print(f"Fail to init bot: {e}")
        return

    # Создаем приложение бота
    app = Application.builder().token(system.config["token"]).build()

    # Регистрируем обработчики команд
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("add", add_camera))
    app.add_handler(CommandHandler("remove", remove_camera))
    app.add_handler(CommandHandler("list", list_cameras))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("snapshot", snapshot))
    app.add_handler(CommandHandler("sensitivity", set_sensitivity))
    app.add_handler(CommandHandler("motion", toggle_motion))
    app.add_handler(CommandHandler("check", check_cameras))

    # Запускаем мониторинг камер в фоне
    asyncio.create_task(system.check_motion())

    # Запускаем бота
    print("Bot started!")
    print(f"ADmin chat ID: {system.config['admin_chat_id']}")
    
    try:
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        
        # Бесконечный цикл
        while True:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Err in main loop: {e}")
    finally:
        await app.stop()
        await app.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nБот остановлен")
    except Exception as e:
        print(f"FATL: {e}")