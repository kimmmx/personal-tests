import csv
import bs4
import time
import urllib
from bs4 import BeautifulSoup
from datetime import timedelta, date
from multiprocessing.pool import ThreadPool


def is_num(val):
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def daterange(start_date, end_date):
    dates = []
    for n in range(int((end_date - start_date).days)):
        dates.append(start_date + timedelta(n))
    return dates


def get_data(some_date):

    # Initialize row set
    row_set = []

    # Extract year, month, and day from date
    date_time = some_date.strftime('%Y-%m-%d')
    year = some_date.strftime('%Y')
    month = some_date.strftime('%m')
    day = some_date.strftime('%d')
    print "Parsing {}...".format(date_time)

    # Loop over stations
    for station in stations:

        # Define url
        url = base_url.format(station, year, month, day)

        # Download webpage
        while True:
            try:
                result = urllib.urlopen(url).read()
                break
            except IOError:
                time.sleep(1)
        soup = BeautifulSoup(result, 'lxml', from_encoding='utf-8')

        # Create list for values
        values = [None for x in value_names]

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
                    if not is_num(value):
                        continue
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
                elif tag.span.text == 'Snow':
                    values[4] = value
                elif tag.span.text == 'Wind Speed':
                    values[5] = value
                elif tag.span.text == 'Max Wind Speed':
                    values[6] = value
                elif tag.span.text == 'Max Gust Speed':
                    values[7] = value

        # Append values to rows
        row_set.append([date_time, station] + values)

    # Return rows
    return row_set


def start_threadpool():

    # Create date range
    date_range = daterange(start, end)

    # Get data
    pool = ThreadPool(4)
    row_sets = pool.map(get_data, date_range)
    pool.close()
    pool.join()

    # Get rows from row_sets and write to csv
    with open(output_csv, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['date_time', 'station'] + value_names)
        for each_set in row_sets:
            writer.writerows(each_set)


if __name__ == '__main__':

    # Start timer
    start_time = time.time()

    # Define output file
    output_csv = 'historic_wunderground_data.csv'

    # Define start and end date
    start = date(2012, 1, 1)
    end = date(2012, 1, 31)  # date.today()

    # Initialize url and stations
    base_url = 'https://www.wunderground.com/history/airport/{0}/{1}/{2}/{3}/DailyHistory.html'
    stations = ['KMDW', 'KORD', 'KUGN', 'KPWK', 'KVYS', 'KPNT', 'KC09', 'KJOT', 'KLOT',
                'KIKK', 'KMLI', 'KSQI', 'KFEP', 'KRFD', 'KRPJ', 'KDKB', 'KARR', 'KDPA']

    # Define values to pull
    value_names = ['avg_temp', 'max_temp', 'min_temp', 'prcp', 'snow', 'avg_wind', 'max_wind', 'max_gust']

    # Run threads to scrape data
    start_threadpool()

    print "Time to complete script: {}".format(str(timedelta(seconds=int(time.time() - start_time))))
