"""Ghép src/shell.html + src/core.js + data/*.json thành web_demo/index.html.

Dữ liệu được NHÚNG THẲNG vào trang thay vì ``fetch``: nhờ vậy mở index.html
bằng cách nhấp đúp (giao thức file://) vẫn chạy, không vướng CORS.

    python docs/sparsemax_mdn_k8/demo/tools/build.py
"""
import io
import json
import os
import re
import sys

# Console Windows mặc định cp1252, không in được tiếng Việt.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHELL = os.path.join(ROOT, 'src', 'shell.html')
CORE = os.path.join(ROOT, 'src', 'core.js')
DATA = os.path.join(ROOT, 'assets')
DST = os.path.join(ROOT, 'index.html')

FETCH_CALL = """Promise.all([fetch('weights.json').then(r=>r.json()),fetch('presets.json').then(r=>r.json())])
.then(([w,p])=>{"""
INLINE_CALL = """Promise.resolve([JSON.parse(document.getElementById('W').textContent),
                 JSON.parse(document.getElementById('P').textContent)])
.then(([w,p])=>{"""


def read(path):
    return io.open(path, encoding='utf-8').read()


def main():
    shell, core = read(SHELL), read(CORE)
    weights, presets = read(os.path.join(DATA, 'weights.json')), read(os.path.join(DATA, 'presets.json'))

    # '</' phải được chẻ ra, nếu không trình duyệt đóng thẻ <script> sớm
    def embed(tag, payload):
        safe = payload.replace('</', '<' + chr(92) + '/')
        return '<script id="' + tag + '" type="application/json">' + safe + '</' + 'script>\n'

    if '/*__CORE__*/' not in shell:
        raise SystemExit('shell.html thiếu dấu /*__CORE__*/')
    page = shell.replace('<script>\n/*__CORE__*/',
                         embed('W', weights) + embed('P', presets) + '<script>\n' + core)
    if FETCH_CALL in page:
        page = page.replace(FETCH_CALL, INLINE_CALL)

    io.open(DST, 'w', encoding='utf-8').write(page)

    # kiểm lại: hai khối JSON phải parse được
    for m in re.finditer(r'<script id="(\w)" type="application/json">(.*?)</script>', page, re.S):
        json.loads(m.group(2).replace('<\\/', '</'))
        print(f'  JSON #{m.group(1)} hợp lệ')
    print(f'index.html = {os.path.getsize(DST) / 1024:.0f} KB')


if __name__ == '__main__':
    main()
