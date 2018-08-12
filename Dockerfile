FROM waleedka/modern-deep-learning

RUN pip install --upgrade pip
RUN pip install pytest

ADD . / /app/

WORKDIR /app
RUN pip install -e .
