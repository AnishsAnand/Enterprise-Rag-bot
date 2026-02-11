# API Specification: kafka - list

**Resource:** kafka
**Operation:** list
**Aliases:** kafka service, kafka services, apache kafka

## Endpoint
- **Method:** POST
- **URL:** https://ipcloud.tatacommunications.com/paasservice/api/v1/paas/listManagedServices/IKSKafka
- **Auth:** Bearer token (from Keycloak)
- **Description:** List Kafka managed services across endpoints

## Required Parameters
- `endpoints`

## Optional Parameters
None

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: list_kafka
List Kafka services across endpoints
- Step 1: get_paas_engagement (engagement.get)
- Step 2: convert_to_ipc_engagement (k8s_cluster.get_ipc_engagement) (depends on: engagement_id)
- Step 3: get_endpoints (endpoint.list) (depends on: paas_engagement_id)
- Step 4: list_kafka_services (.)
