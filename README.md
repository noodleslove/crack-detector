# Crack Detector Web App

A web application that uses computer vision to detect and analyze cracks in concrete structures, providing real-time analysis and reporting capabilities.

## Features

- [x] **Real-time Crack Detection**: Upload images or use camera feed to detect cracks instantly
- [ ]  **Measurement Analysis**: Automatically measure crack width, length, and pattern
- [ ]  **Severity Classification**: AI-powered classification of crack severity levels
- [ ]  **Report Generation**: Generate detailed PDF reports with analysis results
- [ ]  **Historical Tracking**: Monitor crack progression over time
- [ ]  **Mobile Responsive**: Works seamlessly on both desktop and mobile devices

## Tech Stack

- **Frontend**: React.js, Bootstrap
- **Backend**: Python, FastAPI 
- **ML Model**: PyTorch, Hugging Face, Resnet18
- **Database**: PostgreSQL
- **Image Processing**: OpenCV
<!-- - **Cloud Storage**: AWS S3 -->
<!-- - **Authentication**: JWT with OAuth2 -->

## Getting Started

### Prerequisites

<!-- - Node.js >= 16.x -->
- Python >= 3.11
- PostgreSQL >= 13
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/noodleslove/crack-detector.git
cd crack-detector
```

2. Install frontend dependencies:
```bash
cd frontend
npm install
```

3. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
python manage.py init-db
```

### Running the Application

#### Development Mode

1. Start the backend server:
```bash
cd backend
python main.py
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

#### Production Mode

Using Docker:
```bash
docker-compose up -d
```

The application will be available at `http://localhost:3000`

## API Documentation

API endpoints are accessible at `http://localhost:8000/docs` (Swagger UI)

Key endpoints:
- `POST /api/analyze`: Upload and analyze image
- `GET /api/reports`: Retrieve analysis reports
<!-- - `POST /api/calibrate`: Calibrate detection settings -->

## Configuration

Key configuration options in `.env`:
```
DATABASE_URL=postgresql://user:password@localhost/crack_detector
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
MODEL_THRESHOLD=0.5
```

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Submit a pull request

## Testing

Run backend tests:
```bash
cd backend
pytest
```

Run frontend tests:
```bash
cd frontend
npm test
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Security

- All uploads are scanned for malware
- Image metadata is stripped before storage
- Rate limiting is enabled on all API endpoints
- Regular security audits are performed

## Troubleshooting

Common issues and solutions:

1. **Model not loading**
   - Check if model weights are downloaded
   - Verify CUDA compatibility

2. **Upload errors**
   - Verify file size is under 10MB
   - Check supported image formats (JPG, PNG)

3. **Database connection issues**
   - Verify PostgreSQL is running
   - Check database credentials

## Support

For support, please:
1. Check the [FAQ](docs/FAQ.md)
2. Search [existing issues](https://github.com/your-org/crack-detector/issues)
3. Create a new issue if needed

## Acknowledgments

- OpenCV community for image processing tools
- Contributors and maintainers