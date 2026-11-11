"""Persist API key using Windows DPAPI (current Windows user); never plaintext."""
from __future__ import annotations
from pathlib import Path
from core.settings import ROOT
import sys

CRED=ROOT/'config'/'api_key.dpapi'

def _protect(data:bytes,decrypt=False)->bytes:
    if sys.platform!='win32':
        raise RuntimeError('API Key安全持久化仅在Windows上启用；当前系统不保存密钥')
    import ctypes
    from ctypes import wintypes
    class DATA_BLOB(ctypes.Structure):
        _fields_=[('cbData',wintypes.DWORD),('pbData',ctypes.POINTER(ctypes.c_ubyte))]
    crypt=ctypes.windll.crypt32
    kernel=ctypes.windll.kernel32
    buffer=ctypes.create_string_buffer(data,len(data))
    src=DATA_BLOB(len(data),ctypes.cast(buffer,ctypes.POINTER(ctypes.c_ubyte)))
    dst=DATA_BLOB()
    fn=crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    # UI forbidden; encrypted for same user profile
    fn.argtypes=([ctypes.POINTER(DATA_BLOB),ctypes.c_void_p,ctypes.c_void_p,
                  ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(DATA_BLOB)]
                 if not decrypt else [ctypes.POINTER(DATA_BLOB),ctypes.c_void_p,
                  ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(DATA_BLOB)])
    fn.restype=wintypes.BOOL
    if not fn(ctypes.byref(src),None,None,None,None,0x1,ctypes.byref(dst)):
        raise ctypes.WinError()
    try:return ctypes.string_at(dst.pbData,dst.cbData)
    finally:
        kernel.LocalFree.argtypes=[ctypes.c_void_p]
        kernel.LocalFree.restype=ctypes.c_void_p
        kernel.LocalFree(ctypes.cast(dst.pbData,ctypes.c_void_p))

def save_api_key(key:str):
    key=key.strip()
    if not key:raise ValueError('API Key不能为空')
    encrypted=_protect(key.encode('utf-8'))
    CRED.parent.mkdir(parents=True,exist_ok=True)
    CRED.write_bytes(encrypted)

def load_api_key():
    if not CRED.is_file():return ''
    return _protect(CRED.read_bytes(),decrypt=True).decode('utf-8')

def has_api_key():return CRED.is_file()
