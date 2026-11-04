# Flet ZeroShot

To run the development server with hot reload, run the following command:

```sh
uv run flet run --ios -r
```

To build for production for iOS, run the following command:

```sh
uv run flet build ipa -v
```

To check for lint errors, run the following command:

```sh
uv run ruff check .
```

To check for type errors, run the following command:

```sh
uv run pyright
```
