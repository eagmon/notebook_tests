import random


def get_even_numners():
    return [i for i in range(0,100,2)]

def get_odd_numbers():
    return [i for i in range(1,100,2)]

def get_random_numbers():
    return [random.randint(0,100) for i in range(100)]


def main():
    print(get_even_numners())
    print(get_odd_numbers())
    print(get_random_numbers())


if __name__ == "__main__":
    main()
