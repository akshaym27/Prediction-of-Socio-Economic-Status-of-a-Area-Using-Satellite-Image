from owslib.wms import WebMapService
URL = "https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms?request=GetCapabilities"
wms = WebMapService(URL, version='1.1.1')

OUTPUT_DIRECTORY = './data/image_tiles/'

x_min = 90000
y_min = 427000
dx, dy = 200, 200
no_tiles_x = 100
no_tiles_y = 100
total_no_tiles = no_tiles_x * no_tiles_y

x_max = x_min + no_tiles_x * dx
y_max = y_min + no_tiles_y * dy
BOUNDING_BOX = [x_min, y_min, x_max, y_max]

for ii in range(0,no_tiles_x):
    print(ii)
    for jj in range(0,no_tiles_y):
        ll_x_ = x_min + ii*dx
        ll_y_ = y_min + jj*dy
        bbox = (ll_x_, ll_y_, ll_x_ + dx, ll_y_ + dy) 
        img = wms.getmap(layers=['Actueel_ortho25'], srs='EPSG:28992', bbox=bbox, size=(256, 256), format='image/jpeg', transparent=True)
        filename = "{}_{}_{}_{}.jpg".format(bbox[0], bbox[1], bbox[2], bbox[3])
        out = open(OUTPUT_DIRECTORY + filename, 'wb')
        out.write(img.read())
        out.close()
