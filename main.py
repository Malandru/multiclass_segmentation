from examples.numbers import NumClass
import visualize


def main():
    num_class = NumClass()
    num_class.visualize_dataset()
    num_class.run()
    # visualize.show_prepared_figures()


if __name__ == '__main__':
    main()
