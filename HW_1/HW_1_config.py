class Config:
    def __init__(self):
        self._directory = 'Data'

    # Директория для сохранения файлов с данными
    @property
    def Directory(self):
        return self._directory


config = Config()
