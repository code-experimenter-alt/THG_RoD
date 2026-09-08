"""Local corrections on a path-vectorized copy of the original Figure 1.

No generated bitmap is embedded. All unchanged artwork comes from the old
PNG's traced contours, retaining the old panel positions, fonts and colors.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import cairosvg

OUT = Path(__file__).resolve().parent
SVG = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG)
root = ET.Element(f'{{{SVG}}}svg', {'width':'6.455in','height':'2.98866in',
    'viewBox':'0 0 2048 947.929', 'version':'1.1'})
defs = ET.SubElement(root,f'{{{SVG}}}defs')
for name, a, b in [('peach','#fae1d1','#fae1d1'),('blue','#dce8f4','#dce8f4'),
                    ('query','#b9d5e7','#b9d5e7'),('green','#e1eed2','#e1eed2')]:
    g=ET.SubElement(defs,f'{{{SVG}}}linearGradient', {'id':name,'x1':'0','y1':'0','x2':'0','y2':'1'})
    ET.SubElement(g,f'{{{SVG}}}stop',{'offset':'0','stop-color':a})
    ET.SubElement(g,f'{{{SVG}}}stop',{'offset':'1','stop-color':b})
base=ET.parse(OUT/'fig1_old_base.svg').getroot()
group=ET.SubElement(root,f'{{{SVG}}}g',{'id':'original-figure-contours','transform':'scale(0.4739643601)'})
for element in list(base):
    group.append(element)
patches=ET.SubElement(root,f'{{{SVG}}}g',{'id':'local-semantic-corrections'})

def rect(x,y,w,h,fill):
    return ET.SubElement(patches,f'{{{SVG}}}rect',{'x':str(x),'y':str(y),'width':str(w),'height':str(h),'fill':f'url(#{fill})'})

def text(x,y,value,size=33,anchor='middle'):
    e=ET.SubElement(patches,f'{{{SVG}}}text',{'x':str(x),'y':str(y),'font-family':'Liberation Serif,Times New Roman,serif',
        'font-size':str(size),'text-anchor':anchor,'fill':'#080808'})
    e.text=value
    return e

# Correct only the old grammatical note and the supervision-source error.
rect(28,839,429,86,'peach')
text(236,876,'DP protects private',38)
text(236,916,'training records.',38)
rect(690,643,235,82,'query')
text(807,675,'Optional offline',38)
text(807,716,'teacher query',38)
rect(789,740,36,46,'blue')  # Remove teacher-query -> hard-label arrow.
rect(705,787,369,62,'blue')
text(886,812,'Independent hard labels',38)
text(881,839,'y',29)
text(900,844,'aux',20)
# Preserve the two region intervals while fixing the high-threshold equality.
rect(1441,138,220,36,'green')
text(1551,166,'low-health',38)
rect(1458,607,191,48,'green')
text(1472,638,'Ĥ ≥ τ',38,anchor='start')
text(1575,645,'high',25)
payload=ET.tostring(root,encoding='utf-8',xml_declaration=True)
assert b'<image' not in payload and b'data:image' not in payload
(OUT/'fig1_old_vector.svg').write_bytes(payload)
cairosvg.svg2pdf(bytestring=payload,write_to=str(OUT/'fig1_old_vector.pdf'))
print('Original-layout Figure 1 saved as vector SVG/PDF without embedded rasters.')
