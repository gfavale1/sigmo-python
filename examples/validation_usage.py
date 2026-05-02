import sigmo

def main():
    result = sigmo.match(
        query="c1ccccc1",
        target="CCOC(=O)c1ccccc1",
        input_format="smiles",
        iterations=0,
        find_first=True,
        device="auto",
        validate_with_rdkit=True,
    )

    print(result.summary())
    print()
    print(result.explain())


if __name__ == "__main__":
    main()