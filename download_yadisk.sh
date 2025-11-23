#!/bin/bash

if ! command -v yadisk &> /dev/null; then
    echo "yadisk не найден. Установка Ruby и RubyGems..."
    sudo apt-get update
    sudo apt-get install -y ruby-full rubygems
    
    echo "Установка yadisk через gem..."
    sudo gem install yadisk
fi

if [ -z "$1" ]; then
    echo "Использование: $0 <ссылка_на_яндекс_диск> [путь_для_сохранения]"
    echo "Пример: $0 https://yadi.sk/i/HEjuI2Ln3RiRcQ"
    echo "Пример: $0 https://yadi.sk/i/HEjuI2Ln3RiRcQ /path/to/directory"
    echo "Пример: $0 https://yadi.sk/i/HEjuI2Ln3RiRcQ /path/to/directory/file.pth"
    exit 1
fi

YADISK_LINK="$1"
DOWNLOAD_PATH="${2:-.}"

if [[ "$DOWNLOAD_PATH" != */ ]] && [ ! -d "$DOWNLOAD_PATH" ]; then
    TARGET_DIR=$(dirname "$DOWNLOAD_PATH")
    TARGET_FILE=$(basename "$DOWNLOAD_PATH")
    
    if [ "$TARGET_DIR" = "." ]; then
        TARGET_DIR="$(pwd)"
    fi
    
    if [ ! -d "$TARGET_DIR" ]; then
        echo "Создание директории $TARGET_DIR..."
        mkdir -p "$TARGET_DIR"
    fi
    
    echo "Скачивание из $YADISK_LINK в $TARGET_DIR..."
    yadisk "$YADISK_LINK" "$TARGET_DIR"
    
    DOWNLOADED_FILE=$(ls -t "$TARGET_DIR" 2>/dev/null | head -n 1)
    if [ -n "$DOWNLOADED_FILE" ] && [ "$DOWNLOADED_FILE" != "$TARGET_FILE" ] && [ -f "$TARGET_DIR/$DOWNLOADED_FILE" ]; then
        echo "Переименование $DOWNLOADED_FILE в $TARGET_FILE..."
        mv "$TARGET_DIR/$DOWNLOADED_FILE" "$DOWNLOAD_PATH"
    fi
else
    if [ ! -d "$DOWNLOAD_PATH" ]; then
        echo "Создание директории $DOWNLOAD_PATH..."
        mkdir -p "$DOWNLOAD_PATH"
    fi
    
    echo "Скачивание из $YADISK_LINK в $DOWNLOAD_PATH..."
    yadisk "$YADISK_LINK" "$DOWNLOAD_PATH"
fi

echo "Готово!"
