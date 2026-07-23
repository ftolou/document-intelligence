# Thin application image for the receipt web app.
# It intentionally contains only source code on top of a prebuilt runtime image.
# Rebuild this when Python source/static files change; it should be fast.
ARG APP_RUNTIME_IMAGE=receipt-app-runtime:py311
FROM ${APP_RUNTIME_IMAGE}

WORKDIR /app

COPY . /app

ENV PYTHONPATH=/app/src
EXPOSE 7860
CMD ["python", "-m", "receipt_intelligence.app"]
