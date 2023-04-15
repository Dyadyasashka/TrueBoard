FROM python:3.10
EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN apt-get update && apt-get install -y mesa-utils

RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run true_board_app.py
