import aiohttp
import json
import base64
import os
from datetime import datetime
from typing import Dict, Any, List

class PinataStorage:
    def __init__(self):
        self.jwt = os.getenv('PINATA_JWT')
        if not self.jwt:
            raise ValueError("PINATA_JWT environment variable not set")
        self.api_url = "https://api.pinata.cloud/pinning/pinFileToIPFS"

    async def pin_image(self, image_data: str, filename: str = None) -> Dict[str, Any]:
        """
        Pin image to IPFS using Pinata
        
        Args:
            image_data: Base64 encoded image data
            filename: Optional filename
            
        Returns:
            dict: Pinata response containing IpfsHash and other metadata
        """
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"together_ai_{timestamp}.png"

            # Create form data
            data = aiohttp.FormData()
            data.add_field('file',
                         image_bytes,
                         filename=filename,
                         content_type='image/png')

            # Add Pinata metadata
            pinata_metadata = {
                "name": filename,
                "keyvalues": {
                    "source": "together_ai",
                    "timestamp": datetime.now().isoformat()
                }
            }
            data.add_field('pinataMetadata',
                         json.dumps(pinata_metadata),
                         content_type='application/json')

            # Add Pinata options
            pinata_options = {
                "cidVersion": 1
            }
            data.add_field('pinataOptions',
                         json.dumps(pinata_options),
                         content_type='application/json')

            # Make request to Pinata
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.jwt}"
                }
                async with session.post(self.api_url,
                                      data=data,
                                      headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Pinata API error: {error_text}")
                    
                    result = await response.json()
                    return result

        except Exception as e:
            raise