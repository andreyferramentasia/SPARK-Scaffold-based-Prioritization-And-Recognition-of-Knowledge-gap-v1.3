def pytest_addoption(parser):
    parser.addoption(
        "--run-pipeline",
        action="store_true",
        default=False,
        help="Executa o pipeline R antes dos testes de saída",
    )
