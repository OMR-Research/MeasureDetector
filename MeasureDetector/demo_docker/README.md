# Running the server with docker

| ⚠️ WARNING️: The server should not be used for deployment, only testing! |
| --- |

Put a frozen model in this directory and rename it to `model.pb`.

Build the image:

```bash
$ docker build -t measure_detector .
```

Run in container (change port to `XXXX:8080` if needed):

```bash
$ docker run -p 8080:8080 measure_detector
```

## Testing

See `test.py` for a query example.