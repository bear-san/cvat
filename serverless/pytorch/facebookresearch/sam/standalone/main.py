import base64
import cgi
import io
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs
from PIL import Image
from model_handler import ModelHandler

def init_context(context):
    context.logger.info("Init context...  0%")
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("call handler")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = image.convert("RGB")  #  to make sure image comes in RGB
    features = context.user_data.model.handle(image)

    return context.Response(body=json.dumps({
        'blob': base64.b64encode(features.cpu().numpy() if features.is_cuda else features.numpy()).decode(),
    }),
        headers={},
        content_type='application/json',
        status_code=200
    )

class SimpleResponse:
    def __init__(self, body, headers=None, content_type='application/json', status_code=200):
        self.body = body
        self.headers = headers or {}
        self.content_type = content_type
        self.status_code = status_code


class NuclioContext:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
        self.logger = logging.getLogger('cvat-custom-image')
        self.user_data = SimpleNamespace()

    def Response(self, **kwargs):
        return SimpleResponse(**kwargs)


def create_server(host='0.0.0.0', port=8080):
    context = NuclioContext()
    init_context(context)

    class RequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ('/health', '/healthz'):
                self.send_error(404, 'Not Found')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'OK')

        def do_POST(self):
            content_type = self.headers.get('Content-Type', '')
            content_type_lower = content_type.lower()

            length = int(self.headers.get('Content-Length', 0))
            raw_body = self.rfile.read(length)
            payload = None

            if 'multipart/form-data' in content_type_lower:
                form_fp = io.BytesIO(raw_body)
                environ = {
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': content_type,
                }
                form = cgi.FieldStorage(fp=form_fp, headers=self.headers, environ=environ)
                payload = {}
                fields = form.list or []
                for field in fields:
                    if not field.name:
                        continue
                    if field.filename:
                        field.file.seek(0)
                        value = base64.b64encode(field.file.read()).decode('ascii')
                    else:
                        value = field.value

                    existing = payload.get(field.name)
                    if existing is None:
                        payload[field.name] = value
                    elif isinstance(existing, list):
                        existing.append(value)
                    else:
                        payload[field.name] = [existing, value]
            elif 'application/json' in content_type_lower:
                charset = self.headers.get_content_charset() or 'utf-8'
                try:
                    payload = json.loads(raw_body.decode(charset))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self.send_error(400, 'Invalid JSON body')
                    return
                if not isinstance(payload, dict):
                    self.send_error(400, 'JSON body must be an object')
                    return
            elif 'application/x-www-form-urlencoded' in content_type_lower:
                charset = self.headers.get_content_charset() or 'utf-8'
                try:
                    decoded_body = raw_body.decode(charset)
                except UnicodeDecodeError:
                    self.send_error(400, 'Invalid form encoding')
                    return
                params = parse_qs(decoded_body, keep_blank_values=True)
                payload = {}
                for key, values in params.items():
                    if not values:
                        continue
                    if len(values) == 1:
                        payload[key] = values[0]
                    else:
                        payload[key] = values
            else:
                context.logger.info("Unsupported Content-Type: %s", content_type)
                self.send_error(400, 'multipart/form-data, application/x-www-form-urlencoded, or application/json required')
                return

            event = SimpleNamespace(body=payload)

            try:
                response = handler(context, event)
            except Exception:  # noqa: BLE001 - base handler must not crash server
                context.logger.exception('Handler execution failed')
                self.send_error(500, 'Internal server error')
                return

            response_body = response.body.encode('utf-8') if isinstance(response.body, str) else response.body
            self.send_response(response.status_code)

            headers = response.headers or {}
            for header, value in headers.items():
                self.send_header(header, value)

            if response.content_type:
                self.send_header('Content-Type', response.content_type)
            self.send_header('Content-Length', str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format, *args):  # noqa: A003 - follow BaseHTTPRequestHandler signature
            context.logger.info("%s - - %s", self.client_address[0], format % args)

    server = HTTPServer((host, port), RequestHandler)
    context.logger.info('Starting HTTP server on %s:%s', host, port)
    return server


def main():
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '8080'))
    server = create_server(host=host, port=port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
