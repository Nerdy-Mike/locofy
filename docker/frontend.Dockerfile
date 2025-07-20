FROM python:3.9-slim

WORKDIR /app

# Install Stockfish and create symlink
RUN apt-get update && \
  apt-get install -y stockfish && \
  ln -s /usr/games/stockfish /usr/bin/stockfish && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"] 