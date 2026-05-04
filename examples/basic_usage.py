import sigmo

def main():
    """
    Run the simplest possible SIGMo search.

    This example matches the query molecule "CC" against the target molecule
    "CCC" using the high-level sigmo.match() API.
    """
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
        dataframe = result.to_dataframe()
        print()
        print(dataframe)
    except ImportError:
        print()
        print("Pandas is not installed: skipping DataFrame conversion.")


if __name__ == "__main__":
    main()