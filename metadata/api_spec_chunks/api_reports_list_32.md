# API Specification: reports - list

**Resource:** reports
**Operation:** list
**Aliases:** report, reports, common cluster report, common cluster, cluster inventory report, cluster report, cluster inventory, cluster compute report, compute report, cluster compute, storage inventory report, storage report, pvc report

## Endpoint
- **Method:** POST
- **URL:** https://ipcloud.tatacommunications.com/ipcreports/reports/{report_name}
- **Auth:** Bearer token (from Keycloak)
- **Description:** Fetch report data by report name.

## Required Parameters
None

## Optional Parameters
- `report_name`
- `page`
- `size`
- `engagement_id`
- `startDate`
- `endDate`
- `clusterName`
- `datacenter`
- `status`
- `k8sVersion`
- `workerNode`
- `pvcType`

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
No workflow defined