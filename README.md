# FACE RECOGNITION

- Buat virtual environment

```
sudo apt install python3-venv
```

```
python3 -m venv .venv
```

- Install package

```
/home/[path]/.venv/bin/pip install -r requirements.txt
```

- Ganti file `env` menjadi `.env`
- Kemudian ubah isi dari `ACCESS_TOKEN`, ACCESS_TOKEN ini digunakan untuk validasi Authorization Bearer Token yang di kirim dari client.

- Buat script systemd service `faceapi.service`

```
sudo nano /etc/systemd/system/faceapi.service
```

```
[Unit]
Description=Gunicorn Server Face Recognition
After=network.target

[Service]
User=sejator # sesuaikan
Group=www-data # sesuaikan
WorkingDirectory=/home/sejator/path/to
Environment="PATH=/home/sejator/path/to/.venv/bin"
ExecStart=/home/sejator/path/to/.venv/bin/gunicorn --workers 3 --bind unix:faceapi.sock -m 007 wsgi:app

[Install]
WantedBy=multi-user.target
```

- Untuk konfigurasi lebih lengkap mengenai `gunicorn` bisa cek dokumentasinya disini [https://docs.gunicorn.org/en/latest/run.html#commands](https://docs.gunicorn.org/en/latest/run.html#commands)

```
sudo systemctl daemon-reload
```

- Agar otomatis start saat boot

```
sudo systemctl enable faceapi.service
```

- Untuk mulai menjalankan service

```
sudo systemctl start faceapi.service
```

- Cek status

```
sudo systemctl status faceapi.service
```

- Cek report service

```
sudo journalctl -u faceapi.service
```

## Konfigurasi Nginx

[https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-18-04](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-18-04)

```
server {
    listen 80;
    server_name your_domain www.your_domain;

    location / {
        include proxy_params;
        proxy_pass http://unix:/home/sejator/path/to/faceapi.sock;
    }
}
```

## Dokumentasi Postman

[https://documenter.getpostman.com/view/20500330/2sAYJ4jgTv](https://documenter.getpostman.com/view/20500330/2sAYJ4jgTv)
