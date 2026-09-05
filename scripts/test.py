from pygbif import occurrences as occ

# Query occurrences with images
results = occ.search(
    scientificName='Salamandra salamandra',
    mediaType='StillImage',
    limit=100
)

image_urls = []
for record in results['results']:
    for media in record.get('media', []):
        if media.get('type') == 'StillImage':
            image_urls.append(media.get('identifier'))