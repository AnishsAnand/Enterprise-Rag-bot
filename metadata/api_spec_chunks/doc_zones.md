# Zones | Vayu Cloud - Product Documentation

**Resource:** zone
**Operation:** doc
**Aliases:** zone, zones, what is a zone, zone definition, network zone, logical network segment

## Introduction

Zones are logical network segments that provide network connectivity and addressing within an Environment. A Zone refers to a logical network segregated through a firewall, enabling secure communication between virtual machines and network resources.

Within an Environment, VMs are grouped into Zones to provide network isolation and security. When creating a Zone, you can configure network options and subnet allocation.

Zones are logical network segments that provide network connectivity and addressing within an Environment. You can create multiple Zones within an Environment to launch VMs with different network configurations.

In simple terms, Zones act as virtual network containers that enable secure communication between virtual machines while providing network isolation and firewall protection. It is like having separate network rooms within a building, each with its own security rules and access controls.

Zones support both IPv4 and IPv6 network protocols. You can create zones using either IPv4 or IPv6 addressing, or enable the **Dual Stack** option to support both protocols simultaneously. This flexibility allows you to choose the most appropriate network configuration for your specific requirements and future-proof your infrastructure.

## Networking Components

### Network

A network in OpenStack represents a virtual network that provides connectivity between instances and other network resources. It acts as a logical container for subnets and defines the network topology within a zone.

### Subnet

A subnet is a block of IP addresses and associated configuration state within a network. It defines the IP address range, gateway, and DNS settings for the network. Subnets are used to allocate IP addresses when new instances are created and provide network segmentation within zones.

**Key characteristics of subnets:**
- **IP Address Range**: Defines the start and end IP addresses available for allocation
- **Subnet Mask**: Determines the network portion and host portion of IP addresses
- **Gateway**: The default route for traffic leaving the subnet
- **DNS Settings**: Domain name resolution configuration for the subnet
- **DHCP**: Automatic IP address assignment to devices within the subnet
- **Network Segmentation**: Logical separation of network traffic for security and performance

### Port

A port is a connection point for attaching a single device (such as a virtual machine's network interface) to a virtual network. Each port has a unique MAC address and can be assigned one or more IP addresses from the subnet. Ports also define the network configuration and security group associations.

### Firewall

A firewall is a network security device that monitors and controls incoming and outgoing network traffic based on predetermined security rules. In the context of zones, firewalls provide an additional layer of security by filtering traffic between different network segments, protecting virtual machines from unauthorized access and potential threats.

## Create Zone

1. Navigate to the **ZONES** page
2. Click the **+ CREATE ZONE** icon
3. Populate: Environment Name, Zone Name, Purpose (IPC or NFV), Firewall, Hypervisor Choice (e.g. KVM), Zone Type (OVERLAY or VLAN)
4. Production IP Subnet: Select **Auto IPAM** for automatic IP address management
5. Number of IPs: Enter the number of IP addresses needed. The Subnet Mask is automatically set (e.g. /27) and is editable.
6. **Dual Stack**: Enable to support both IPv4 and IPv6 addressing for the zone
7. Click **SAVE & DEPLOY**

## Edit Zone

Click the **pencil icon** (edit) on the desired zone. Update the required fields (such as Zone Name, Number of IPs, Dual Stack, etc.). Save the changes to update the zone configuration.

## Delete Zone

Click the **delete icon** (trash bin) on the desired zone. Confirm the deletion in the dialog box. The zone will be removed from the environment.

## Dual Stack

Dual Stack is a networking technology that allows a zone to support both IPv4 and IPv6 protocols simultaneously. This means your zone can communicate using both the traditional IPv4 addressing and the newer IPv6 addressing, providing enhanced network capabilities and future-proofing your infrastructure.

IPv6 cannot be created separately - it can only be created along with IPv4.

### Enable Dual Stack

When creating or editing a zone, set the Dual Stack option to **Yes**. The system will ask for the number of IPv6 IPs required for the zone.

### Edit Dual Stack

Click the **pencil icon** (edit) on the desired IPv4 zone. Set the Dual Stack option to **Yes**. The system will ask for the number of IPv6 IPs you want to allocate. Save the changes. The zone will now be configured for both IPv4 and IPv6.

---

Source: https://ipcloud.tatacommunications.com/docs/docs/zones
