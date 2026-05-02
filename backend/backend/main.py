from __future__ import annotations

import base64
import json
import mimetypes
import os
import pathlib
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / 'demo_static'
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '').strip()
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash').strip() or 'gemini-1.5-flash'
PORT = int(os.getenv('PORT', '8000'))


def json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode('utf-8')


def read_file(path: pathlib.Path) -> bytes:
    return path.read_bytes()


def gemini_analyze(prompt_text: str, image_bytes: bytes | None, mime_type: str | None) -> str:
    if not GEMINI_API_KEY:
        return 'Gemini is not configured. Set GEMINI_API_KEY in your environment to enable this feature.'

    url = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}'
    parts = [{'text': prompt_text}]
    if image_bytes is not None:
        parts.append({
            'inline_data': {
                'mime_type': mime_type or 'image/png',
                'data': base64.b64encode(image_bytes).decode('ascii'),
            }
        })

    payload = {'contents': [{'role': 'user', 'parts': parts}]}
    request = urllib.request.Request(
        url,
        data=json_bytes(payload),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as error:
        details = error.read().decode('utf-8', errors='ignore') if hasattr(error, 'read') else str(error)
        raise RuntimeError(f'Gemini request failed: {details}') from error
    except Exception as error:
        raise RuntimeError(f'Gemini request failed: {error}') from error

    candidates = body.get('candidates', [])
    if not candidates:
        return 'Gemini returned no candidates.'

    content = candidates[0].get('content', {})
    parts = content.get('parts', [])
    texts = [part.get('text', '') for part in parts if isinstance(part, dict)]
    response_text = '\n'.join(texts).strip()
    return response_text or 'Gemini returned an empty response.'


class DemoHandler(BaseHTTPRequestHandler):
    server_version = 'SegmentationDemo/1.0'

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json_bytes(payload)
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: pathlib.Path, content_type: str | None = None) -> None:
        body = read_file(path)
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type or (mimetypes.guess_type(str(path))[0] or 'application/octet-stream'))
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ('/', '/index.html'):
            index_path = STATIC_DIR / 'index.html'
            if not index_path.exists():
                self._send_json(HTTPStatus.NOT_FOUND, {'detail': 'Demo not found'})
                return
            self._send_file(index_path, 'text/html; charset=utf-8')
            return

        if self.path == '/status':
            self._send_json(HTTPStatus.OK, {
                'server_mode': 'stdlib',
                'mock_prediction': True,
                'gemini_enabled': bool(GEMINI_API_KEY),
                'gemini_model': GEMINI_MODEL,
            })
            return

        if self.path.startswith('/static/'):
            rel_path = self.path.removeprefix('/static/').lstrip('/')
            file_path = STATIC_DIR / rel_path
            if file_path.exists() and file_path.is_file():
                self._send_file(file_path)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {'detail': 'Static file not found'})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {'detail': 'Not found'})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == '/predict':
            self._send_json(HTTPStatus.NOT_IMPLEMENTED, {
                'detail': 'This lightweight server keeps prediction in the browser fallback mode.',
            })
            return

        if self.path != '/gemini/analyze':
            self._send_json(HTTPStatus.NOT_FOUND, {'detail': 'Not found'})
            return

        content_type = self.headers.get('Content-Type', '')
        if 'application/json' not in content_type:
            self._send_json(HTTPStatus.BAD_REQUEST, {'detail': 'Expected application/json'})
            return

        content_length = int(self.headers.get('Content-Length', '0'))
        try:
            request_body = json.loads(self.rfile.read(content_length).decode('utf-8'))
        except Exception:
            self._send_json(HTTPStatus.BAD_REQUEST, {'detail': 'Invalid JSON body'})
            return

        prompt_value = request_body.get('prompt', 'Describe this off-road scene and mention any terrain risks.')
        image_data_url = request_body.get('image_data_url')
        image_bytes = None
        mime_type = request_body.get('mime_type') or 'image/png'

        if isinstance(image_data_url, str) and image_data_url.startswith('data:') and ',' in image_data_url:
            header, encoded = image_data_url.split(',', 1)
            if ';base64' in header:
                try:
                    image_bytes = base64.b64decode(encoded)
                    if header.startswith('data:') and ';' in header:
                        mime_type = header[5:].split(';', 1)[0] or mime_type
                except Exception:
                    self._send_json(HTTPStatus.BAD_REQUEST, {'detail': 'Invalid image_data_url'})
                    return

        try:
            response_text = gemini_analyze(prompt_value, image_bytes, mime_type)
        except Exception as error:
            self._send_json(HTTPStatus.BAD_GATEWAY, {'detail': str(error)})
            return

        self._send_json(HTTPStatus.OK, {
            'response': response_text,
            'enabled': bool(GEMINI_API_KEY),
            'model': GEMINI_MODEL,
        })

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        print(f'{self.address_string()} - {format % args}')


def main() -> None:
    if not STATIC_DIR.exists():
        raise SystemExit(f'Static directory not found: {STATIC_DIR}')

    server = ThreadingHTTPServer(('127.0.0.1', PORT), DemoHandler)
    print(f'Starting server on http://127.0.0.1:{PORT}')
    print('Serving demo_static/ and Gemini proxy endpoint')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Stopping server...')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()