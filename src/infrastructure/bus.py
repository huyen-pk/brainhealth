import pika

class RabbitMqClient:
    def __init__(self, host='localhost', queue='default'):
        self.host = host
        self.queue = queue
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue)

    def publish(self, message):
        self.channel.basic_publish(exchange='', routing_key=self.queue, body=message)
        print(f" [x] Sent '{message}'")

    def consume(self, callback):
        def on_message(channel, method, properties, body):
            callback(body)
            channel.basic_ack(delivery_tag=method.delivery_tag)

        self.channel.basic_consume(queue=self.queue, on_message_callback=on_message)
        print(' [*] Waiting for messages. To exit press CTRL+C')
        self.channel.start_consuming()

    def close(self):
        self.connection.close()

# Example usage:
# client = RabbitMqClient(host='localhost', queue='test')
# client.publish('Hello, World!')
# client.consume(lambda msg: print(f"Received: {msg}"))
class MessagingQueue:
    def __init__(self, client):
        self.client = client

    def publish(self, message):
        self.client.publish(message)

    def consume(self, callback):
        self.client.consume(callback)

    def close(self):
        self.client.close()

# Example usage:
# rabbitmq_client = RabbitMqClient(host='localhost', queue='test')
# messaging_queue = MessagingQueue(rabbitmq_client)
# messaging_queue.publish('Hello, World!')
# messaging_queue.consume(lambda msg: print(f"Received: {msg}"))