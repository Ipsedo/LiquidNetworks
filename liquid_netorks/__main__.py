# -*- coding: utf-8 -*-
import argparse

from .dummy import identity


def main() -> None:
    parser = argparse.ArgumentParser("liquid_netorks main")

    parser.add_argument("message", type=str, help="The message to display")

    args = parser.parse_args()

    message = identity(args.message)

    print(message)


if __name__ == "__main__":
    main()
