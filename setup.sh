# Установка зависимостей
pip install -r requirements.txt

# Запуск приложения
streamlit run app.py

mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"galex77777777@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
" > ~/.streamlit/config.toml