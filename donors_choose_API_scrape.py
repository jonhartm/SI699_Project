from datetime import datetime, timedelta

from caching import *
from secrets import *

API_URL = 'https://api.donorschoose.org/common/json_feed.html'

def check_string_for_csv(s):
    return '"{}"'.format(s) if ',' in s else s

def parse_proposal(proposal_data):
    return [
        proposal_data['id'],
        proposal_data['latitude'],
        proposal_data['longitude'],
        proposal_data['city'],
        proposal_data['state'],
        proposal_data['zip'],
        check_string_for_csv(proposal_data['subject']['name']),
        check_string_for_csv(proposal_data['resource']['name']),
        proposal_data['gradeLevel']['name'],
        proposal_data['povertyType']['label'],
        proposal_data['totalPrice'],
        proposal_data['numStudents'],
        proposal_data['numDonors'],
        proposal_data['fundingStatus'],
        proposal_data['expirationDate'],
        str(datetime.strptime(proposal_data['fullyFundedDate'][:-4], '%A, %B %d, %Y %I:%M:%S %p'))
    ]
    if 'additionalSubjects' in proposal_data:
        for add_subject in proposal_data['additionalSubjects']:
            print(add_subject['name'])


def get_projects_by_id(proposal_ids):
    projects = []

    # max of 50 projects per request, so split them up
    for id_range in range(0, len(proposal_ids), 50):
        print("Requesting proposals {} thru {}...".format(id_range,id_range+50))
        params = {
            "APIKey":DONORS_CHOOSE_API_KEY,
            "max":"50",
            "id":','.join(proposal_ids[id_range:id_range+50])
        }
        API_data = Cache.CheckCache_API(
            API_URL,
            params,
            rate_limit=1
        )

        for project in API_data['proposals']:
            projects.append(project)
    return projects


def get_project_ids_by_date(date):
    params = {
        "APIKey":DONORS_CHOOSE_API_KEY,
        "historical":"true",
        "newSince":str(dt.timestamp()*1000)[:-2],
        "olderThan":str((dt+timedelta(days=1)).timestamp()*1000)[:-2],
        "concise":"true"
    }
    API_data = Cache.CheckCache_API(
        API_URL,
        params,
        rate_limit=1
    )
    print("Found {} proposals for {}".format(API_data['totalProposals'], date))

    return [p['id'] for p in API_data['proposals']]

dt = datetime(year=2016, month=10, day=1)
cache_filename = "cache_{:02d}_{}.json".format(dt.month, dt.year)
Cache = CacheFile(cache_filename)
with open("output.csv", 'w') as f:
    f.write("id,lat,long,city,state,zip,primay_subject,resource,gradelevel,povertylevel,totalPrice,numStudents,numDonors,fundingStatus,expirationDate,fullyFundedDate\n")
    while dt < datetime.now():
    # while dt < datetime(year=2017, month=10, day=31):
        dt = dt + timedelta(days=1)
        if cache_filename != "cache_{:02d}_{}.json".format(dt.month, dt.year):
            cache_filename = "cache_{:02d}_{}.json".format(dt.month, dt.year)
            Cache = CacheFile(cache_filename)
            print("Starting a new cache file...", cache_filename)

        ids = get_project_ids_by_date(dt)
        for project in get_projects_by_id(ids):
            f.write(','.join(parse_proposal(project)) + '\n')
