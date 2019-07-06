import shapefile
import json

input_filename = './nwb_wegvakken/Wegvakken.shp'
output_filename = './nwb_wegvakken/Wegvakken.json'

reader = shapefile.Reader(shp_filename)
fields = reader.fields[1:]
field_names = [field[0] for field in fields]
buffer = []
for sr in reader.shapeRecords():
    atr = dict(zip(field_names, sr.record))
    geom = sr.shape.__geo_interface__
    buffer.append(dict(type="Feature", geometry=geom, properties=atr)) 

output_filename = './data/nwb_wegvakken/2017_09_wegvakken.json'
json_file = open(output_filename , "w")
json_file.write(json.dumps({"type": "FeatureCollection", "features": buffer}, indent=2, default=JSONencoder) + "\n")
json_file.close()
