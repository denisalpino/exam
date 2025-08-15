FROM ollama/ollama:0.9.6

# 1) Устанавливаем curl (опционально, для будущих healthchecks)
RUN apt-get update \
    && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint_embed.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

EXPOSE 11434
