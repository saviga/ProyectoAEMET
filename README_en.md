**🌍  Temperature Prediction API and Gemini Assistant for AEMET** 

Welcome to the repository for the Temperature Prediction API and Gemini Assistant. This project is a FastAPI-based web service that combines a hybrid Machine Learning model to predict temperature and an integrated virtual assistant for historical data queries.

The API is deployed on an AWS EC2 instance and uses PostgreSQL as the database to store historical weather data. The ML models are loaded from an S3 bucket for efficient resource management.

**🚀 Key Features**

    -Temperature Prediction: Uses a hybrid model that combines a stack of bidirectional LSTM layers with a stack of Transformer blocks, trained with PyTorch, to generate future temperature forecasts.

    -Smart Assistant: Integrates a conversational assistant (based on Gemini) that can answer questions about the historical data available in the database.

    -Robust Architecture: The ML models are loaded at application startup from AWS S3, ensuring the service is always ready to respond without delays.

    -RESTful API: Offers clear RESTful endpoints for forecast prediction and for interacting with the assistant.

**🛠️Prerequisites**

To run this project, you need to have the following components installed:

    -Python 3.9+

    -PostgreSQL (with a database configured and historical data)

    -EC2 (with a .pem key for SSH connection, and a security group with inbound and outbound traffic permissions configured for the ports to be used)

    -RDS (configured for necessary traffic)

    -AWS Access (with credentials configured for S3)

    -Create Lambdas with the LambdaS3 (to extract data from the API and save it to S3) and LambdaBD (to save the extracted data to RDS) files.

        <remember to set the environment variables in the lambda configuration and the necessary layers for the libraries>
  	

**⚙️Installation and Configuration**

Follow these steps to get the project up and running in your local environment or on an EC2 instance.

**1. Clone the Repository**

git clone <repository_url>
cd <your_repository_name>

**2. Grant Permissions to Your .pem Key to Allow SSH Connection to EC2**

chmod 400 /path_to_your_pem_key/key_name.pem
   
**3. SSH Connection to Your EC2 Instance**

ssh -v -i '/path_to_your_pem_key/key_name.pem' ec2-user@<ec2_public_ip>

    <I recommend setting your elastic IP so your IP is not volatile>

    <ec2-user changes depending on the AMI (Amazon Machine Image) you use to launch the EC2 instance>

**4. Configure the Virtual Environment and Dependencies**

sudo yum update -y
sudo yum install -y python3 python3-pip git nginx awscli
python3 -m venv venv
source venv/bin/activate

**5. Create a Directory**

mkdir /home/ec2-user/fastapi_app
cd /home/ec2-user/fastapi_app

**6. Download Content from GitHub or S3 and Install Dependencies**

    A. Copy the necessary files directly from your GitHub repository

        <We create an empty repository for selective download>
        git clone --no-checkout <repository_URL>
        cd <repository_name>
        git sparse-checkout init --cone

        <We download the API content>
        git sparse-checkout set API
        git sparse-checkout set API/Main_production.py API/model_architecture.py API/model_production.py API/qa_production.py API/requirements.txt

        <We clear the current sparse-checkout configuration>
        git sparse-checkout set --no-cone

        <We download the model and its weights from the streamlit directory>
        git sparse-checkout set streamlit
        git sparse-checkout set streamlit/production_weather_model.pth streamlit/scaler_X_production.joblib streamlit/scaler_y_production.joblib

        <To avoid download errors, we reset sparse-checkout again>
        git sparse-checkout set --no-cone

        <Dependency installation>
        pip install -r requirements.txt

    B. Create an S3 Bucket in your AWS console and upload the contents of the API folder in addition to the production_weather_model.pth, scaler_X_production.joblib, and scaler_y_production.joblib files found in the streamlit folder.

        <Make sure your EC2 instance has the necessary permissions to access S3>

        <To do this, go to IAM in AWS and create a new role with the following policy>
                        {
                        "Version": "2012-10-17",
                        "Statement": [
                                    {
                                    "Effect": "Allow",
                                    "Action": [
                                    "s3:GetObject",
                                    "s3:ListBucket"
                                    ],
                                    "Resource": [
                                    "arn:aws:s3:::<your_bucket_name>",
                                    "arn:aws:s3:::<your_bucket_name>/*"
                                                                    ]
                                                                        }
                                                                        ]
                                                                        }

        <After creating the role, assign it to your EC2 instance>

        <Download files from S3 to EC2>
        aws s3 sync s3://<path_to_your_file>/file_name/ .

        <Dependency installation>
        pip install -r requirements.txt

**7. Configure Environment Variables**

The project uses environment variables to connect to the PostgreSQL database. You must define them in your terminal before starting the application.

Replace the example values with your actual credentials:

export GOOGLE_API_KEY=<api_key>
export PG_HOST="<your_postgresql_host>"
export PG_PORT="5432" # or the port you use
export PG_USER="<your_postgresql_user>"
export PG_PASSWORD="<your_postgresql_password>"
export PG_DATABASE="<your_postgresql_database>"


**8. Start the Application**

Once the environment variables are configured, you can start the Uvicorn server. It is important to use the --workers 1 flag for model loading.

uvicorn Main_production:app --host 0.0.0.0 --port 8000 --workers 1

    #If everything is configured correctly, you will see in the console that the models are loaded from S3 and that the application starts without errors.

**🧪 API Usage**

Once the API is running, you can interact with it through two main endpoints: Forecast (for predictions) and Ask (to receive historical information).

    From Streamlit App:
    https://proyecto-final-hack-a-bossgit-wx2inpdg5kukgh6xymcwkh.streamlit.app/

    From the Console:

        Endpoint 1: Temperature Prediction

            This endpoint takes a location and a number of days to predict the temperature.

            Route: /forecast

            Method: POST

            Payload: JSON with the fields ubicacion (string) and dias (integer).

        Example usage with curl:
        curl -X POST "http://<EC2_PUBLIC_IP>:8000/forecast" \
        -H "Content-Type: application/json" \
        -d '{"ubicacion": "Madrid", "dias": 5}'

        Expected response (JSON):
        {
        "ubicacion": "Madrid",
        "pronostico": [
        { "fecha": "2025-08-08", "temperatura": 25.5 },
        { "fecha": "2025-08-09", "temperatura": 26.1 },
        { "fecha": "2025-08-10", "temperatura": 25.9 },
        { "fecha": "2025-08-11", "temperatura": 26.3 },
        { "fecha": "2025-08-12", "temperatura": 26.8 }
        ]
        }

        Endpoint 2: Query Assistant

            This endpoint allows you to ask questions about historical data.

            Route: /ask

            Method: POST

            Payload: JSON with the question (string) field.

        Example usage with curl:
        curl -X POST "http://<EC2_PUBLIC_IP>:8000/ask" \
        -H "Content-Type: application/json" \
        -d '{"pregunta": "What was the average temperature in Madrid in May 2024?"}'

        Expected response (JSON):
        {
        "respuesta": "The average temperature in Madrid in May 2024 was 21.5 degrees Celsius.",
        "sql_generada": "SELECT AVG(tmed) FROM datos_clima WHERE nombre = 'Madrid' AND fecha BETWEEN '2024-05-01' AND '2024-05-31'"
        }

**🤝 Contributions**

Contributions are welcome. If you have ideas for improvement, open an issue or create a pull request.

**📖 Documentation**

You can access the interactive API documentation in your browser by visiting the following address:

http://localhost:8000/docs

This will take you directly to the Swagger UI interface, where you can explore and test all of your API's endpoints.
