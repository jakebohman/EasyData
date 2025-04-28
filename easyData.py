import pandas as pd
import glob as glob
import numpy as np

as2name = {7922: 'Comcast', 14593:'Starlink', 395354: 'Starry', 20057: 'AT&T Mobility', 11427: 'Charter (TX)', 11426: 'Charter (Carolinas)', 20115: 'Charter (20115)',  10796: 'Charter (Midwest)',6167: 'Verizon Wireless',5650: 'Frontier',209: 'CenturyLink',
           7029: 'Windstream', 701: 'Verizon Business', 204285: 'Optimum', 6181: 'Cincinnati Bell',30036: 'Mediacom', 36149: 'Hawaiian Telecom',7018: 'AT&T Services', 6697: 'Beltelecom', 2856: 'British Telecom', 396356: 'Latitude', 5378: 'Vodafone UK',
           13335: 'Cloudflare', 21928 : 'T-Mobile', 4713 : 'NTT', 22773 : 'Cox', 55836 : 'Jio', 9299 : 'Philippine LDT', 24560 : 'Bharti Airtel', 8151 : 'Uninet', 16509 : 'Amazon', 396982 : 'Google', 5089 : 'Virgin Media', 2516 : 'KDDI', 9829 : "NIB India",
           3209 : 'Vodafone GER', 4766 : 'Korea Telecom', 2527 : 'Sony',3320 : 'Deutsche Telekom', 22616 : 'ZSCALER', 33363 : 'Charter (BHN)', 20001 : 'Charter (PACWEST)', 7155:'ViaSat',6621:'Hughesnet'}
ASNsAll = [7922, 7018, 13335, 21928, 4713, 22773, 55836, 701, 20115, 9299, 6167, 2516, 8151, 5089, 396982, 14593, 16509, 24560, 10796, 2856, 11427, 9829, 3209, 209, 4766, 2527, 3320, 22616, 33363, 20001]
ASNs1_10 = [7922, 7018, 13335, 21928, 4713, 22773, 55836, 701, 20115, 9299]
ASNs11_20 = [6167, 2516, 8151, 5089, 396982, 14593, 16509, 24560, 10796, 2856]
ASNs21_30 = [11427, 9829, 3209, 209, 4766, 2527, 3320, 22616, 33363, 20001]
ISPsShared = ['CenturyLink','Comcast','Charter','Verizon','Cox','Verizon DSL']
ISPsAll = ['NA','CenturyLink','Comcast','Frontier','Charter','Verizon','Windstream','Cincinnati Bell','Optimum','Cox','Mediacom','Verizon DSL','Hawaiian Telcom']
ISPsall = ['NA','CenturyLink','Comcast','Frontier','Charter','Verizon','Windstream','Cincinnati Bell','Optimum','Cox','Mediacom','Hawaiian Telcom']

def load_jsons(path):
    dfs = []
    files = glob.glob(path)
    for file in files:
        if(".json" in file):
            tempdata = pd.read_json(file, lines=True)
            dfs.append(tempdata)
    data = pd.concat(dfs, ignore_index = True)
    return data

def load_MBA(path):
    dfs = []
    dfs.append(pd.read_csv("/local/nw-latency/mba/202212/" + path))
    dfs.append(pd.read_csv("/local/nw-latency/mba/202301/" + path))
    dfs.append(pd.read_csv("/local/nw-latency/mba/202302/" + path))
    dfs.append(pd.read_csv("/local/nw-latency/mba/202304/" + path))
    dfs.append(pd.read_csv("/local/nw-latency/mba/202305/" + path))
    dfs.append(pd.read_csv("/local/nw-latency/mba/202306/" + path))
    dfs.append(pd.read_csv("/local/nw-latency/mba/202307/" + path))
    data = pd.concat(dfs, ignore_index=True)
    unit_mappings = pd.read_csv("/local/nw-latency/mba/unit_mappings.csv")
    dictionary = dict(zip(unit_mappings['Unit ID'], unit_mappings['ISP']))
    data['ISP'] = data['unit_id'].map(dictionary)
    data['ISP'].fillna("NA",inplace=True)
    return data