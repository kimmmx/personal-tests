import urllib
import bs4
from datetime import timedelta, date
from bs4 import BeautifulSoup
from tqdm import tqdm


def daterange(start_date, end_date):
    dates = []
    for n in range(int((end_date - start_date).days)):
        dates.append(start_date + timedelta(n))
    return dates


def main():

    # Define start and end date
    start_date = date(2012, 1, 1)
    end_date = date.today()

    # Initialize url and stations
    base_url = 'https://www.wunderground.com/history/airport/{0}/{1}/{2}/{3}/DailyHistory.html'
    stations = ['KMDW', 'KORD', 'KUGN', 'KPWK', 'KVYS', 'KPNT', 'KC09', 'KJOT', 'KLOT',
                'KIKK', 'KMLI', 'KSQI', 'KFEP', 'KRFD', 'KRPJ', 'KDKB', 'KARR', 'KDPA']

    # Loop over date range
    for each_date in tqdm(daterange(start_date, end_date)):
        year = each_date.strftime('%Y')
        month = each_date.strftime('%m')
        day = each_date.strftime('%d')

        # Loop over stations
        for station in stations:
            url = base_url.format(station, year, month, day)
            result = urllib.urlopen(url).read()
            soup = BeautifulSoup(result, 'lxml', from_encoding='utf-8')

            # Create list for values
            values = [None for x in ['avg_temp', 'max_temp', 'min_temp', 'prcp', 'avg_wind', 'max_wind', 'max_gust']]

            # Find relevant tags in history table
            tags = []
            table = soup.find(id='historyTable')
            for tr in table.tbody:
                if type(tr) == bs4.element.Tag:
                    tags.append(tr)

            # Loop through tags
            for tag in tags:
                if tag.span:
                    try:
                        value = tag(attrs={'class': 'wx-value'})[0].text
                    except IndexError:
                        continue

                    if tag.span.text == 'Mean Temperature':
                        values[0] = value
                    elif tag.span.text == 'Max Temperature':
                        values[1] = value
                    elif tag.span.text == 'Min Temperature':
                        values[2] = value
                    elif tag.span.text == 'Precipitation':
                        values[3] = value
                    elif tag.span.text == 'Wind Speed':
                        values[4] = value
                    elif tag.span.text == 'Max Wind Speed':
                        values[5] = value
                    elif tag.span.text == 'Max Gust Speed':
                        values[6] = value


if __name__ == '__main__':
    main()
