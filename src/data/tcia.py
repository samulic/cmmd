import urllib.request
import urllib.parse
import sys, os
#
# Refer https://wiki.cancerimagingarchive.net/display/Public/REST+API+Usage+Guide for complete list of API
# source: github.com/sharmalab/tcia-sdk

class TCIAClient:
    GET_IMAGE = "getImage"
    GET_SINGLE_IMAGE = "getSingleImage"
    GET_MANUFACTURER_VALUES = "getManufacturerValues"
    GET_MODALITY_VALUES = "getModalityValues"
    GET_COLLECTION_VALUES = "getCollectionValues"
    GET_BODY_PART_VALUES = "getBodyPartValues"
    GET_PATIENT_STUDY = "getPatientStudy"
    GET_SERIES = "getSeries"
    GET_PATIENT = "getPatient"    
    NEW_STUDIES_IN_PATIENT_COLLECTION = "NewStudiesInPatientCollection"
    GET_SOP_INSTANCE_UIDS = "getSOPInstanceUIDs"
    PATIENTS_BY_MODALITY = "PatientsByModality"
    GET_SERIES_SIZE = "getSeriesSize"
    NEW_PATIENTS_IN_COLLECTION = "NewPatientsInCollection"
    GET_SHARED_LIST = "getSharedList"


    def __init__(self, credentials , baseUrl):
        self.credentials = credentials
        self.baseUrl = baseUrl
        
    def execute(self, url, queryParameters={}):
        queryParameters = dict((k, v) for k, v in queryParameters.items() if v)
        credentialsHeader = "ldap" + " " + self.credentials
        headers = {"Authorization" : credentialsHeader}
        queryString = "?%s" % urllib.parse.urlencode(queryParameters)
        requestUrl = url + queryString
        request = urllib.request.Request(url=requestUrl , headers=headers)
        resp = urllib.request.urlopen(request)
        return resp
    
    def get_modality_values(self,collection = None , bodyPartExamined = None, outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/" + self.GET_MODALITY_VALUES
        queryParameters = {"Collection" : collection , "BodyPartExamined" : bodyPartExamined , "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp
    
    def get_manufacturer_values(self,collection = None , bodyPartExamined = None , modality = None , outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/" + self.GET_MANUFACTURER_VALUES
        queryParameters = {"Collection" : collection , "BodyPartExamined" : bodyPartExamined , "Modality" : modality , "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp
        
    def get_collection_values(self,outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/" + self.GET_COLLECTION_VALUES
        queryParameters = { "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp
        
    def get_body_part_values(self,collection = None , modality = None , outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/" + self.GET_BODY_PART_VALUES
        queryParameters = {"Collection" : collection , "Modality" : modality , "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp

    def get_patient_study(self,collection = None , patientId = None , studyInstanceUid = None , outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/" + self.GET_PATIENT_STUDY
        queryParameters = {"Collection" : collection , "PatientID" : patientId , "StudyInstanceUID" : studyInstanceUid , "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp

    def get_series(self, collection = None, studyInstanceUid = None, modality = None, patientID = None, seriesInstanceUid = None, bodyPartExamined = None, manufacturer = None, manufacturerModelName = None, outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/" + self.GET_SERIES
        queryParameters = {"Collection" : collection , "StudyInstanceUID" : studyInstanceUid , "Modality" : modality , "SeriesInstanceUID": seriesInstanceUid, "BodyPartExamined": bodyPartExamined, "Manufacturer": manufacturer, "ManufacturerModelName": manufacturerModelName, "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp

    def get_patient(self, collection = None , outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/" + self.GET_PATIENT
        queryParameters = {"Collection" : collection , "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp

    def get_image(self, seriesInstanceUid):   # SeriesInstanceUID: required
        serviceUrl = self.baseUrl + "/" + self.GET_IMAGE
        queryParameters = { "SeriesInstanceUID" : seriesInstanceUid }
        resp = self.execute( serviceUrl , queryParameters)
        return resp

    def get_image(self , seriesInstanceUid , downloadPath, zipFileName):
        serviceUrl = self.baseUrl + "/" + self.GET_IMAGE
        queryParameters = { "SeriesInstanceUID" : seriesInstanceUid }
        os.umask(0o002)
        try:
            file = os.path.join(downloadPath, zipFileName)
            resp = self.execute(serviceUrl , queryParameters)
            downloaded = 0
            CHUNK = 256 * 10240
            with open(file, 'wb') as fp:
                while True:
                    chunk = resp.read(CHUNK)
                    downloaded += len(chunk)
                    if not chunk: 
                        break
                    fp.write(chunk)
        except urllib.error.HTTPError as e:
            print("HTTP Error:",e.code , serviceUrl)
            return False
        except urllib.error.URLError as e:
            print("URL Error:",e.reason , serviceUrl)
            return False

        return True

    def get_single_image(self, sopInstanceUid, seriesInstanceUid):   # SeriesInstanceUID: required, sopInstanceUID: required
        serviceUrl = self.baseUrl + "/" + self.GET_SINGLE_IMAGE
        queryParameters = { "SOPInstanceUID" : sopInstanceUid, "SeriesInstanceUID" : seriesInstanceUid}        
        resp = self.execute( serviceUrl , queryParameters)
        return resp        
    
    def new_studies_in_patient_collection(self, collection, date, patientId = None, outputFormat = "json"):   # collection: required, date: required
        serviceUrl = self.baseUrl + "/" + self.NEW_STUDIES_IN_PATIENT_COLLECTION
        queryParameters = { "Collection" : collection, "Date" : date, "PatientID": patientId, "format" : outputFormat}
        resp = self.execute( serviceUrl , queryParameters)
        return resp


    def get_sop_instance_uids(self, seriesInstanceUid, outputFormat = "json"):   # SeriesInstanceUID: required
        serviceUrl = self.baseUrl + "/" + self.GET_SOP_INSTANCE_UIDS
        queryParameters = { "SeriesInstanceUID" : seriesInstanceUid, "format" : outputFormat}
        resp = self.execute( serviceUrl , queryParameters)
        return resp

                
    def patients_by_modality(self, collection, modality, outputFormat = "json"):   # collection: required, modality: required
        serviceUrl = self.baseUrl + "/" + self.PATIENTS_BY_MODALITY
        queryParameters = { "Collection" : collection, "Modality" : modality, "format" : outputFormat}
        resp = self.execute( serviceUrl , queryParameters)
        return resp


    def get_series_size(self, seriesInstanceUid, outputFormat = "json"):   # SeriesInstanceUID: required
        serviceUrl = self.baseUrl + "/" + self.GET_SERIES_SIZE
        queryParameters = { "SeriesInstanceUID" : seriesInstanceUid, "format" : outputFormat}
        resp = self.execute( serviceUrl , queryParameters)
        return resp

     
    def new_patients_in_collection(self, collection, date, outputFormat = "json"):   # collection: required, date: required
        serviceUrl = self.baseUrl + "/" + self.NEW_PATIENTS_IN_COLLECTION
        queryParameters = { "Collection" : collection, "Date" : date, "format" : outputFormat}
        resp = self.execute( serviceUrl , queryParameters)
        return resp

       
    def get_shared_list(self, name, outputFormat = "json"):   # name: required
        serviceUrl = self.baseUrl + "/" + self.GET_SHARED_LIST
        queryParameters = { "name" : name, "format" : outputFormat}
        resp = self.execute( serviceUrl , queryParameters)
        return resp

def printServerResponse(method, response):
    if response.getcode() == 200:
        print (method + ": The server returned:\n")
        print (response.read())
        print ("\n")
    
    else:
        print ("Error : " + str(response.getcode)) # print error code
