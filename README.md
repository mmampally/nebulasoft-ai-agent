##  Docker Deployment

### Build the Docker Image
```bash
docker build -t nebulasoft-agent .
```

### Run with Docker
```bash
docker run -it --env OPENROUTER_API_KEY=your_key_here nebulasoft-agent
```

### Run with Docker Compose
```bash
# Make sure .env file exists with your API key
echo "OPENROUTER_API_KEY=your_key_here" > .env

# Start the container
docker-compose up
```

### Stop the Container
```bash
# Press Ctrl+C, then:
docker-compose down
```

### Ticket Logs

Tickets are saved in `tickets.log` which persists outside the container.

## Docker Image Details

- **Base Image:** python:3.11-slim
- **Size:** ~1.5GB (includes dependencies and embeddings)
- **Vector DB:** Pre-built during image build
- **API Key:** Passed as environment variable at runtime

