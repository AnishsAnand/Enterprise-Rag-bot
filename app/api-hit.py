import requests

url = "https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken"
data =  {
	"password": "Tata@1234",
	"email": "izo_cloud_admin@tatacommunications.onmicrosoft.com"
}

response = requests.post(url, json=data)
print(response.json())