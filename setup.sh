# Установка зависимостей
sudo apt-get install -y libgl1-mesa-glx
pip install  libgl1-mesa-glx
pip install -r requirements.txt
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:$LD_LIBRARY_PATH

# Запуск приложения
streamlit run true_board_app.py

