"""
QR Code Generator for Material Passports
"""
import qrcode
import io
import base64
import os


def generate_qr_code(data_url: str, component_id: str, save_path: str = None) -> str:
    """
    Generate a QR code for the material passport.
    Returns base64-encoded PNG image.
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(data_url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Save to file if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)

    # Return base64 encoded image
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return img_base64

