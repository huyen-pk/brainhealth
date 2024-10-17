# Configure the AWS Provider
provider "aws" {
    region     = env("AWS_REGION")
    access_key = env("AWS_ACCESS_KEY")
    secret_key = env("AWS_SECRET_KEY")
}

# Configure the Google Cloud Provider
provider "google" {
    project     = env("GCP_PROJECT_ID")
    region      = env("GCP_REGION")
    credentials = file(env("GCP_CREDENTIALS_FILE"))
}

# Configure the Azure Provider
provider "azurerm" {
    features {}
    client_id       = env("AZURE_CLIENT_ID")
    client_secret   = env("AZURE_CLIENT_SECRET")
    subscription_id = env("AZURE_SUBSCRIPTION_ID")
    tenant_id       = env("AZURE_TENANT_ID")
    tenant_id       = env("AZURE_TENANT_ID")
}