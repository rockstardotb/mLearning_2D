FROM waleedka/modern-deep-learning

RUN pip install --upgrade pip
RUN pip install pytest

WORKDIR /app/mLearning_2D/
RUN pip install -e .
