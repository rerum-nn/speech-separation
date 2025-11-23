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
    exit 1
fi

YADISK_LINK="$1"
DOWNLOAD_PATH="${2:-.}"

extract_zip() {
    local file="$1"
    local target_dir="$2"
    
    if [[ "$file" == *.zip ]]; then
        echo "Распаковка ZIP архива: $file"
        if ! command -v unzip &> /dev/null; then
            echo "Установка unzip..."
            sudo apt-get install -y unzip
        fi
        unzip -q "$file" -d "$target_dir" 2>/dev/null || unzip -q "$file"
        echo "Архив $file распакован"
        return 0
    fi
    return 1
}

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
    
    if [ -n "$DOWNLOADED_FILE" ] && [ -f "$TARGET_DIR/$DOWNLOADED_FILE" ]; then
        FULL_DOWNLOADED_PATH="$TARGET_DIR/$DOWNLOADED_FILE"
        
        if extract_zip "$FULL_DOWNLOADED_PATH" "$TARGET_DIR"; then
            rm -f "$FULL_DOWNLOADED_PATH"
            
            DOWNLOADED_ITEM=$(ls -td "$TARGET_DIR"/*/ 2>/dev/null | head -n 1)
            if [ -n "$DOWNLOADED_ITEM" ]; then
                DOWNLOADED_ITEM="${DOWNLOADED_ITEM%/}"
            else
                DOWNLOADED_ITEM=$(ls -t "$TARGET_DIR"/* 2>/dev/null | head -n 1)
            fi
            
            if [ -n "$DOWNLOADED_ITEM" ]; then
                DOWNLOADED_FILE=$(basename "$DOWNLOADED_ITEM")
                FULL_DOWNLOADED_PATH="$DOWNLOADED_ITEM"
            fi
        fi
        
        if [ -n "$DOWNLOADED_FILE" ] && [ "$DOWNLOADED_FILE" != "$TARGET_FILE" ]; then
            if [ -e "$FULL_DOWNLOADED_PATH" ]; then
                echo "Переименование $DOWNLOADED_FILE в $TARGET_FILE..."
                mv "$FULL_DOWNLOADED_PATH" "$DOWNLOAD_PATH"
            fi
        fi
    fi
else
    TARGET_DIR="$DOWNLOAD_PATH"
    TARGET_DIR_NAME=$(basename "$TARGET_DIR")
    
    TEMP_DIR=$(dirname "$TARGET_DIR")
    if [ "$TEMP_DIR" = "." ]; then
        TEMP_DIR="$(pwd)"
    fi
    
    if [ ! -d "$TEMP_DIR" ]; then
        echo "Создание директории $TEMP_DIR..."
        mkdir -p "$TEMP_DIR"
    fi
    
    echo "Скачивание из $YADISK_LINK в $TEMP_DIR..."
    yadisk "$YADISK_LINK" "$TEMP_DIR"
    
    # Находим последний скачанный файл
    DOWNLOADED_FILE=$(ls -t "$TEMP_DIR" 2>/dev/null | head -n 1)
    
    if [ -n "$DOWNLOADED_FILE" ] && [ -f "$TEMP_DIR/$DOWNLOADED_FILE" ]; then
        FULL_DOWNLOADED_PATH="$TEMP_DIR/$DOWNLOADED_FILE"
        
        if extract_zip "$FULL_DOWNLOADED_PATH" "$TEMP_DIR"; then
            rm -f "$FULL_DOWNLOADED_PATH"
            
            DOWNLOADED_DIR=$(ls -td "$TEMP_DIR"/*/ 2>/dev/null | head -n 1)
            if [ -n "$DOWNLOADED_DIR" ]; then
                DOWNLOADED_DIR="${DOWNLOADED_DIR%/}"
            fi
            
            if [ -n "$DOWNLOADED_DIR" ] && [ -d "$DOWNLOADED_DIR" ]; then
                DOWNLOADED_NAME=$(basename "$DOWNLOADED_DIR")
                
                if [ "$DOWNLOADED_DIR" != "$TARGET_DIR" ]; then
                    echo "Переименование $DOWNLOADED_NAME в $TARGET_DIR_NAME..."
                    mv "$DOWNLOADED_DIR" "$TARGET_DIR"
                fi
            fi
        fi
    fi
fi

echo "Готово!"

