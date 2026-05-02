import sigmo

def main():
    result = sigmo.match(
        query="CC",
        target="CCC",
        input_format="smiles",
        iterations=0,
        find_first=True,
        device="auto",
    )

    print(result.summary())
    print()
    print(result.explain())

    try:
        df = result.to_dataframe()
        print()
        print(df)
    except ImportError:
        print()
        print("Pandas non installato: salto la conversione in DataFrame.")


if __name__ == "__main__":
    main()