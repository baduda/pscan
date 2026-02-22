from pscan.main import main
import matplotlib

if __name__ == "__main__":
    # Используем Agg бэкэнд для неинтерактивного запуска, если нужно
    matplotlib.use('Agg')
    main()
