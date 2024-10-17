provider "aws" {
    region = var.aws_region
}

terraform {
    backend "s3" {
        bucket         = "your-terraform-state-bucket"
        key            = "path/to/your/terraform.tfstate"
        region         = var.aws_region
        dynamodb_table = "your-lock-table"
    }
}

resource "aws_s3_bucket" "model_bucket" {
    bucket = "ai-model-training-bucket"
    acl    = "private"
}

resource "aws_s3_bucket_object" "training_data" {
    bucket = aws_s3_bucket.model_bucket.bucket
    key    = "training-data/"
    source = "path/to/your/training/data"
}

resource "aws_iam_role" "tpu_role" {
    name = "tpu-role"

    assume_role_policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
            {
                Action = "sts:AssumeRole"
                Effect = "Allow"
                Principal = {
                    Service = "ec2.amazonaws.com"
                }
            }
        ]
    })
}

resource "aws_iam_role_policy" "tpu_policy" {
    name = "tpu-policy"
    role = aws_iam_role.tpu_role.id

    policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
            {
                Action = [
                    "s3:GetObject",
                    "s3:PutObject"
                ]
                Effect   = "Allow"
                Resource = "${aws_s3_bucket.model_bucket.arn}/*"
            }
        ]
    })
}

resource "aws_instance" "tpu_instance" {
    ami           = "ami-0abcdef1234567890" # Replace with a valid TPU-compatible AMI
    instance_type = "p3.2xlarge" # Example instance type for TPU

    iam_instance_profile = aws_iam_instance_profile.tpu_instance_profile.name

    tags = {
        Name = "TPUInstance"
    }
}

resource "aws_iam_instance_profile" "tpu_instance_profile" {
    name = "tpu-instance-profile"
    role = aws_iam_role.tpu_role.name
}