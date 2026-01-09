import { ChangeDetectorRef, Component, OnInit, TemplateRef, ViewChild, inject } from '@angular/core';
import { FormBuilder, FormControl, FormGroup, Validators } from '@angular/forms';
import { MatStepper } from '@angular/material/stepper';
import { StepperSelectionEvent } from '@angular/cdk/stepper';
import { ClusterService } from '../../services/cluster.service';
import { HostListener } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { ToastrService } from 'ngx-toastr';
import { MatDialog } from '@angular/material/dialog';
import { VmService } from 'src/app/ipc/vm/vm.service';
import { AuditLogDialogComponent } from 'src/app/ipc/common/audit-log-dialog/audit-log-dialog.component';
import { BsModalService } from 'ngx-bootstrap/modal';
import { LoadingMessageTemplateComponent } from 'src/app/ipc/common/components/loading-message-template/loading-message-template.component';
import { DataSourceDetails } from 'src/app/ipc/vm/launch/launch-vm.model';
import { Observable, Subscription, debounceTime, distinctUntilChanged, forkJoin, map, Subject } from 'rxjs';
import { IPCSharedService } from 'src/app/common/services/ipc-shared.services';
import { ClusterSharedService } from '../../services/cluster-shared.service';
import { SnackbarMessageComponent } from 'src/app/common/components/snackbar-message/snackbar-message.component';
import { Router } from '@angular/router';
import { IPCCommonService } from 'src/app/ipc/common/ipc-common.service';
import { CreateServiceComponent } from 'src/app/common/components/create-service/create-service.component';
import { NgIf } from '@angular/common';

interface FlagsObj {
  // displayLoader: boolean;
  // displaySuccessContent: boolean;
  isEngineerUser: boolean;
  isEngineer: boolean;
  isAdmin: boolean;
  isCustomer: boolean;
}

@Component({
    selector: 'app-create-cluster',
    templateUrl: './create-cluster.component.html',
    styleUrls: ['./create-cluster.component.css'],
    standalone: true,
    imports: [NgIf,CreateServiceComponent]
})
export class CreateClusterComponent {
  @ViewChild('createServiceComponent') createServiceComponent: any;
  currentStepper = 0
  flags = <FlagsObj>{};
  selectedDataCenters = [];
  selectedKubernetesVersions = [];
  environments: any[] = [];
  businessUnitOptions: any[] = [];
  environmentOptions: any[] = [];
  zoneslist: any[] = [];
  flavours: any[] = [];
  selectedOS: any
  selectedOSVersion: string = ''; // Store the selected OS version for delayed processing
  selectedBusinessUnitName: string = '';
  selectedEnvironmentName: string = '';
  selectedDropDownValue: string = '';
  selectedEngId: any;
  zoneId: any;
  modalDialog: any = {};
  showSpinner: boolean = false;

  public dataSource = <DataSourceDetails>{};

  // Settings for the multi-select dropdown
  dropdownSettings = {
    singleSelection: false,
    text: "Select Data Centers",
    selectAllText: 'Select All',
    unSelectAllText: 'UnSelect All',
    enableSearchFilter: true,
    classes: "myclass custom-class"
  };

  selectedClusterType: string;
  selectedControlPlaneType: string;
  clusterType: string = ''; // Add this property to store cluster type
  stepDefinitions: any = [];
  engagements: any = [];
  kubeversions: any = [];
  tagslist: any[] = [];
  ostype: any[] = [];
  osOptions: any[] = [];
  flavourOptions: { id: number; itemName: any; }[];
  nodeTypeOptions: { id: number; itemName: string; originalValue: string }[] = []; // Add this property to store node type options
  originalFlavorCategories: string[] = []; // Store original flavor categories for API calls
  selectedNodeType: string = ''; // Add this property to store selected node type
  selectedNodeTypeOriginal: string = ''; // Store original node type value for API calls
  CustEngId: number;
  iksImages: any[] = []; // Add this property to store images
  networks: any;
  selectedDataCenterEndpointId: number | null = null; // Add this property to store selected data center endpointId
  selectedDataCenterEndpointMap: string | null = null; // Add this property to store selected data center endpointmap
  globalOstypeResponse: any = null; // Add this property to store getostype response globally
  isSelectedDataCenterVCPEnabled: boolean = false; // Track if selected data center is VCP-enabled
  originalResponseData: any = null; // Store original response structure to check VCP-Enabled status
  selectedDataCenterCategoryStatus: { inVksEnabled: boolean; inAllImages: boolean } = { inVksEnabled: false, inAllImages: false }; // Track which categories the selected data center is in
  private currentAddonsRequestId: number = 0; // Track the current addons request ID
  private currentNetworkRequestId: number = 0; // Track the current network request ID
  private lastKubernetesVersion: string = ''; // Track the last kubernetes version to prevent duplicate calls

  // Tags functionality properties
  selectedTagKey: any = null;
  keyValueTags: { key: string; value: string; id?: any }[] = [];
  lastSelectedTag: any = null;
  searchInputValues: { [key: string]: string } = {};
  filteredOptions: { [key: string]: any[] } = {};
  response: any[];

  // Helper method to get copfId from global response
  getCopfId(): string {
    return this.globalOstypeResponse?.data?.coffArrayList?.[0]?.copfId || "E-IPCTEAM-1602";
  }



  constructor(private clusterService: ClusterService, private snackBar: MatSnackBar, private toastr: ToastrService,
    private dialog: MatDialog, private vmService: VmService, public bsModalService: BsModalService, private ipcSharedService: IPCSharedService,
    private clusterSharedService: ClusterSharedService, private cdr: ChangeDetectorRef,private router: Router, private ipcCommonService: IPCCommonService) {
  }
  @HostListener('window:focus', [])
  ngOnInit(): void {
    this.flags.isEngineer = this.clusterSharedService.onCheckIsEngineer();
    this.flags.isAdmin = this.ipcSharedService.getSuperAdminForIPC();
    this.flags.isCustomer = this.clusterSharedService.onCheckIsCustomer();

    console.log("USer TYPE", this.flags)
    if (this.engagements.length == 0) {
      if (this.flags.isEngineer) {
      this.getclusterTemplate();
      }
      else{
        this.getclusterTemplatecust();
      }
    this.getEngagements();
    }
  }
  private _formBuilder = inject(FormBuilder);

  firstFormGroup: FormGroup = this._formBuilder.group({ firstCtrl: [''] });
  secondFormGroup: FormGroup = this._formBuilder.group({ secondCtrl: [''] });
  thirdFormGroup: FormGroup = this._formBuilder.group({ thirdCtrl: [''] });
  fourthFormGroup: FormGroup = this._formBuilder.group({ fourthCtrl: [''] });
  fifthFormGroup: FormGroup = this._formBuilder.group({ fifthCtrl: [''] });
  sixthFormGroup: FormGroup = this._formBuilder.group({ fifthCtrl: [''] });

  private callEngagementDependentData(engId: string): void {
    console.log("------------------callEngagementDependentData", engId);

    if (!engId) return;
    this.getikstemplates();
    // this.getkubernetesversion();
    this.getzones();
    this.gettagslist();
    // Call getenvironment without endpointmap initially - it will be called again when datacenter is selected
    // this.getenvironment(); // Commented out - only call when datacenter is selected
    // this.tryFetchProjectList();
  }

  getEngagements() {

    this.clusterService.getEngagements(
      (response) => {
        this.engagements = response.data;
        const formattedVersions = this.engagements.map((data, index) => ({
          id: data.id,
          itemName: data.engagementName
        }));

        this.stepDefinitions.forEach((step) => {
          step.formControls.forEach((control) => {
            if (control.name === 'engagements') {
              control.options = formattedVersions;
            }
          });
        });
                  // Set selectedEngId AFTER engagements are loaded
          if (this.engagements.length > 0 && this.flags.isCustomer) {
            this.CustEngId = this.ipcSharedService.getEngagementId() 
            this.ipcCommonService.getIpcEngFromPaasEng(this.CustEngId,
              (response) => {
                // console.log("response for ipc migration", response);
                this.selectedEngId = response.data.ipc_engid;
                console.log(this.selectedEngId,"selectedEngId")
                // Call this AFTER selectedEngId is set
                this.callEngagementDependentData(this.selectedEngId);
              },
              (error) => {
                console.log("error", error);
              }
            );
          }
      },
      (error) => {
        console.log("error", error);
      }
    );
  }
  // Helper method to check which categories a data center exists in
  private getDataCenterCategoryStatus(endpointId: number): { inVksEnabled: boolean; inAllImages: boolean } {
    const result = { inVksEnabled: false, inAllImages: false };
    
    if (!this.originalResponseData) {
      return result;
    }

    // Handle both array and object response structures
    let categoriesToProcess: any[] = [];
    
    if (Array.isArray(this.originalResponseData)) {
      categoriesToProcess = this.originalResponseData;
    } else if (typeof this.originalResponseData === 'object') {
      // If it's an object, wrap it in an array
      categoriesToProcess = [this.originalResponseData];
    } else {
      console.warn("Unexpected original response data structure:", typeof this.originalResponseData);
      return result;
    }

    // Check which categories the endpointId exists in
    for (const category of categoriesToProcess) {
      if (category && typeof category === 'object') {
        // Check vks-enabledImages category
        const vksEnabledImages = category['vks-enabledImages'];
        if (Array.isArray(vksEnabledImages)) {
          const hasVksImage = vksEnabledImages.some((img: any) => 
            img && img.endpointId === endpointId
          );
          if (hasVksImage) {
            result.inVksEnabled = true;
          }
        }
        
        // Check all-images category
        const allImages = category['all-images'];
        if (Array.isArray(allImages)) {
          const hasAllImage = allImages.some((img: any) => 
            img && img.endpointId === endpointId
          );
          if (hasAllImage) {
            result.inAllImages = true;
          }
        }
      }
    }
    
    return result;
  }

  // Helper method to check if a data center is VCP-enabled (for backward compatibility)
  private isDataCenterVCPEnabled(endpointId: number): boolean {
    const status = this.getDataCenterCategoryStatus(endpointId);
    return status.inVksEnabled;
  }

  // Helper method to extract and flatten images from the new response structure
  private extractImagesFromResponse(responseData: any): any[] {
    const allImages: any[] = [];
    
    if (!responseData) {
      console.warn("No response data available for image extraction");
      return allImages;
    }

    // Handle both array and object response structures
    let categoriesToProcess: any[] = [];
    
    if (Array.isArray(responseData)) {
      categoriesToProcess = responseData;
    } else if (typeof responseData === 'object') {
      // If it's an object, wrap it in an array
      categoriesToProcess = [responseData];
    } else {
      console.warn("Unexpected response data structure:", typeof responseData);
      return allImages;
    }

    // Process each category (vks-enabledImages, all-images, etc.)
    categoriesToProcess.forEach(category => {
      if (typeof category === 'object' && category !== null) {
        Object.keys(category).forEach(categoryName => {
          const categoryImages = category[categoryName];
          if (Array.isArray(categoryImages)) {
            // Add all images from this category, ensuring they have required properties
            const validImages = categoryImages.filter(img => 
              img && 
              typeof img === 'object' && 
              img.ImageName && 
              img.endpointId && 
              img.endpointName && 
              img.endpoint
            );
            allImages.push(...validImages);
            console.log(`Extracted ${validImages.length} valid images from category: ${categoryName}`);
          }
        });
      }
    });

    console.log(`Total images extracted: ${allImages.length}`);
    return allImages;
  }

  getikstemplates() {
    const payload = {
      endpoint_reference: "Ubuntu_22.04"
    };

    this.clusterService.getIksImagesVersion(
      this.selectedEngId,
              (response) => {

          // Handle new response structure with vks-enabledImages and all-images categories
          // The response now contains categories like "vks-enabledImages" and "all-images" 
          // instead of a flat array of images
          // Mock data for testing the new response structure
          // const response = {
          //   "status": "success",
          //   "data": {
          //     "vks-enabledImages": [
          //       {
          //         "ImageName": "UBUNTU22.04_STD_IKS_01JUL2025-v1.26.15",
          //         "endpoint": "EP_V2_UKHB",
          //         "endpointId": 10,
          //         "endpointName": "Highbridge",
          //         "id": 46634
          //       },
          //       {
          //         "ImageName": "ubuntu-2204--APR25-kvm--v1.27.16",
          //         "endpoint": "EP_V2_UKHB",
          //         "endpointId": 10,
          //         "endpointName": "Highbridge",
          //         "id": 46792
          //       },
          //       {
          //         "ImageName": "ubuntu-2204--IKS-AUG25--v1.27.16",
          //         "endpoint": "EP_V2_UKHB",
          //         "endpointId": 10,
          //         "endpointName": "Highbridge",
          //         "id": 46844
          //       }
          //     ],
          //     "all-images": [
          //       {
          //         "ImageName": "ubuntu-2204--IKS-APR25--v1.30.14",
          //         "endpoint": "EP_V2_UKHB",
          //         "endpointId": 10,
          //         "endpointName": "Highbridge",
          //         "id": 46864
          //       },
          //       {
          //         "ImageName": "UBUNTU24.04_STD_IKS_AUG2025-v1.28.15",
          //         "endpoint": "EP_V2_UKHB",
          //         "endpointId": 10,
          //         "endpointName": "Highbridge",
          //         "id": 47403
          //       },
          //       {
          //         "ImageName": "UBUNTU24.04_STD_IKS_AUG25-v1.31.13",
          //         "endpoint": "EP_V2_UKHB",
          //         "endpointId": 10,
          //         "endpointName": "Highbridge",
          //         "id": 47406
          //       },
          //       {
          //         "ImageName": "UBUNTU24.04_STD_IKS_01AUG2025-v1.32.9",
          //         "endpoint": "EP_V2_UKHB",
          //         "endpointId": 10,
          //         "endpointName": "Highbridge",
          //         "id": 47466
          //       },
          //       {
          //         "ImageName": "UBUNTU24.04_STD_IKS_01AUG2025-v1.28.15",
          //         "endpoint": "EP_V2_UKHB",
          //         "endpointId": 10,
          //         "endpointName": "Highbridge",
          //         "id": 47484
          //       },
          //       {
          //         "ImageName": "UBUNTU24.04_STD_IKS_01AUG2025-v1.31.13",
          //         "endpoint": "EP_V2_UKHB",
          //         "endpointId": 10,
          //         "endpointName": "Highbridge",
          //         "id": 47582
          //       },
          //       {
          //         "ImageName": "UBUNTU24.04_STD_IKS_01AUG2025-v1.30.14",
          //         "endpoint": "EP_V2_UKHB",
          //         "endpointId": 10,
          //         "endpointName": "Highbridge",
          //         "id": 47597
          //       },
          //       {
          //         "ImageName": "UBUNTU24.04_STD_IKS_01AUG2025-v1.29.14",
          //         "endpoint": "EP_V2_UKHB",
          //         "endpointId": 10,
          //         "endpointName": "Highbridge",
          //         "id": 47598
          //       },
          //       {
          //         "ImageName": "UBUNTU24.04_STD_IKS_01AUG2025-v1.34.1",
          //         "endpoint": "EP_V2_UKHB",
          //         "endpointId": 10,
          //         "endpointName": "Highbridge",
          //         "id": 47599
          //       },
          //       {
          //         "ImageName": "UBUNTU24.04_STD_IKS_01AUG2025-v1.33.5",
          //         "endpoint": "EP_V2_UKHB",
          //         "endpointId": 10,
          //         "endpointName": "Highbridge",
          //         "id": 47600
          //       },
          //       {
          //         "ImageName": "UBUNTU24.04_STD_IKS_01AUG2025-v1.27.16",
          //         "endpoint": "EP_V2_UKHB",
          //         "endpointId": 10,
          //         "endpointName": "Highbridge",
          //         "id": 47608
          //       },
          //       {
          //         "ImageName": "UBUNTU22.04_STD_IKS_01JAN2025-v1.26.15_VCD",
          //         "endpoint": "EP_V2_DEL",
          //         "endpointId": 11,
          //         "endpointName": "Delhi",
          //         "id": 43126
          //       },
          //       {
          //         "ImageName": "UBUNTU22.04_STD_IKS_01JAN2025-v1.27.16_VCD",
          //         "endpoint": "EP_V2_DEL",
          //         "endpointId": 11,
          //         "endpointName": "Delhi",
          //         "id": 43280
          //       }
          //     ]
          //   },
          //   "message": null,
          //   "responseCode": 0
          // };
          
          // Store original response data for VCP-Enabled checking
          this.originalResponseData = response.data || [];
          
          // Use mock data instead of actual API response for testing
          this.iksImages = this.extractImagesFromResponse(response.data || []);

          
          console.log("Stored IKS images:", this.iksImages);
          
          // Populate Data Center options first (before kubernetes step)
          this.populateDataCenterOptions();
          
          // Call getiksimageversions after images are stored
          // this.getiksimageversions();
        },
      (error) => {
        console.error("Error fetching IKS image versions:", error);
      }
    );
  }

  getiksimageversions(endpointId?: number) {
    // Use stored images from getikstemplates instead of making another API call
    if (this.iksImages && this.iksImages.length > 0) {
      // console.log("Using stored IKS images:", this.iksImages);
      
      // Filter images by selected data center endpointId if provided
      let filteredImages = this.iksImages;
      if (endpointId) {
        filteredImages = this.iksImages.filter(img => img.endpointId === endpointId);
        // console.log(`Filtered images for endpointId ${endpointId}:`, filteredImages.length, "images");
      }
      
      // Step 1: Extract version strings like "v1.27.16" from ImageName
      const versionSet = new Set<string>();
      filteredImages.forEach((img: any) => {
        const match = img.ImageName?.match(/v\d+\.\d+\.\d+/);
        if (match) {
          versionSet.add(match[0]);
        }
      });

      // Step 2: Sort semantically by version number (latest first)
      const uniqueVersions = Array.from(versionSet).sort((a, b) => {
        const parse = (v: string) => v.replace('v', '').split('.').map(Number);
        const [aMajor, aMinor, aPatch] = parse(a);
        const [bMajor, bMinor, bPatch] = parse(b);

        if (bMajor !== aMajor) return bMajor - aMajor;
        if (bMinor !== aMinor) return bMinor - aMinor;
        return bPatch - aPatch;
      });

      // Step 3: Format for dropdown use
      const formattedVersions = uniqueVersions.map((version, index) => ({
        id: version,
        itemName: version
      }));

      // Step 4: Inject into stepDefinitions
      this.stepDefinitions.forEach((step) => {
        step.formControls.forEach((control) => {
          if (control.name === 'kubernetesVersion') {
            control.options = formattedVersions;
          }
        });
      });

      // console.log("📦 Injected Kubernetes Versions:", formattedVersions);
    } else {
      console.warn("No stored IKS images available. Make sure getikstemplates() is called first.");
    }
   
  }

createtags(tags: any[]) {
  this.clusterService.createMultipleTags(tags, (response) => {
    console.log("response", response);
    // Show success message
    let snackbarData = { 
      message: `Successfully created ${tags.length} tag(s).`, 
      type: 'success' 
    };
    this.showSnackbar(snackbarData);
    
    // Refresh tags list after successful creation
    this.gettagslist();
    
    // Notify create-service component that tag creation is complete
    this.createServiceComponent?.onTagCreationComplete(true);
  }, (error) => {
    console.log("error", error);
    // Show error message
    let snackbarData = { 
      message: `Error creating tags. Please try again.`, 
      type: 'error' 
    };
    this.showSnackbar(snackbarData);
    
    // Notify create-service component that tag creation failed
    this.createServiceComponent?.onTagCreationComplete(false);
  });
}

  checktagname(tagName: string, defaultEngId: number) {
    this.clusterService.checkTagNameService(tagName, defaultEngId, (response) => {
      console.log("response", response);
    }, (error) => {
      console.log("error", error);
    });
  }

  gettagslist() {
    this.clusterService.getTagsList(
      this.selectedEngId,
      (response) => {
        this.tagslist = response.data;

        // Map tags to include itemName, id, and description
        const tagOptions = this.tagslist.map(tag => ({
          itemName: tag.name,
          id: tag.id,
          description: tag.description
        }));

        // Inject into the Tags field in stepDefinitions
        this.stepDefinitions?.forEach((step) => {
          step.formControls.forEach((control) => {
            if (control.name === 'Tags') {
              control.options = tagOptions;
            }
          });
        });
      },
      (error) => {
        console.log("error", error);
      }
    );
  }

  getenvironment(endpointmap?: string) {
    this.clusterService.getEnvironmentListPerEngagementService(
      this.selectedEngId,
      (response) => {
        // console.log("environments", response);
        this.environments = response.data.environments || [];

        // Filter environments based on endpointmap if provided
        let filteredEnvironments = this.environments;
        if (endpointmap) {
          filteredEnvironments = this.environments.filter(env => 
            env.endpointName === endpointmap
          );
          console.log(`Filtered environments for endpointmap ${endpointmap}:`, filteredEnvironments.length, "environments");
        }

        const businessUnitMap = new Map();
        filteredEnvironments.forEach(env => {
          if (!businessUnitMap.has(env.departmentId)) {
            businessUnitMap.set(env.departmentId, {
              id: env.departmentId,
              itemName: env.department
            });
          }
        });
        this.businessUnitOptions = Array.from(businessUnitMap.values());

        this.environmentOptions = filteredEnvironments.map(env => ({
          id: env.name,
          envId: env.id,
          itemName: env.name,
          departmentId: env.departmentId
        }));

        this.updateNetworkSetupDropdowns();
      },
      (error) => {
        console.log("error", error);
      }
    );
  }

  getzones() {

    if (this.selectedEngId) {
      this.clusterService.getZoneList(
        this.selectedEngId,
        (response) => {
          console.log("response zones list", response);
          this.zoneslist = response.data;
          // console.log("zoneslist", this.zoneslist);
        },
        (error) => {
          console.log("error", error);
        }
      );
    }
  }

  // getostype() {
  //   this.clusterService.getppuEnabledImages(
  //     this.zoneId,
  //     null,
  //     false,
  //     (response) => {
  //       this.ostype = response.data.image.options || [];

  //       if (!this.kubernetesVersion) {
  //         console.warn("Kubernetes version not available.");
  //         return;
  //       }

  //       // Group images by osMake
  //       const groupedByMake = this.ostype.reduce((acc, img) => {
  //         if (!img.label?.includes(this.kubernetesVersion)) return acc;

  //         if (!acc[img.osMake]) acc[img.osMake] = [];
  //         acc[img.osMake].push(img);
  //         return acc;
  //       }, {} as Record<string, any[]>);

  //       const osOptionsMap = new Map<string, any>();

  //       for (const osMake in groupedByMake) {
  //         const images = groupedByMake[osMake];

  //         images.sort((a, b) => {
  //           const dateA = extractDateFromLabel(a.label);
  //           const dateB = extractDateFromLabel(b.label);
  //           return dateB.getTime() - dateA.getTime(); // Latest first
  //         });

  //         const latestImage = images[0];
  //         if (latestImage) {
  //           osOptionsMap.set(osMake, {
  //             id: latestImage.id,
  //             itemName: osMake,
  //             osModel: latestImage.osModel,
  //             osMake: latestImage.osMake
  //           });
  //         }
  //       }

  //       this.osOptions = Array.from(osOptionsMap.values());

  //       // Set options in stepDefinitions
  //       const workerNodeStep = this.stepDefinitions.find(step => step.label.includes('Worker Node Pool'));
  //       if (workerNodeStep) {
  //         const osControl = workerNodeStep.formControls.find(ctrl => ctrl.name === 'operatingSystem');
  //         if (osControl) {
  //           osControl.options = this.osOptions;
  //         }
  //       }
  //     },
  //     (error) => {
  //       console.log("error", error);
  //     }
  //   );

  //   function extractDateFromLabel(label: string): Date {
  //     const monthMap = {
  //       JAN: 0, FEB: 1, MAR: 2, APR: 3, MAY: 4, JUN: 5,
  //       JUL: 6, AUG: 7, SEP: 8, OCT: 9, NOV: 10, DEC: 11
  //     };

  //     const dateRegex = /(\d{2})([A-Z]{3})(\d{4})/i;
  //     const altRegex = /([A-Z]{3})(\d{4})/i;

  //     let day = 1, month = 0, year = 1970;

  //     const match = label.match(dateRegex);
  //     if (match) {
  //       day = +match[1];
  //       month = monthMap[match[2].toUpperCase()] || 0;
  //       year = +match[3];
  //     } else {
  //       const altMatch = label.match(altRegex);
  //       if (altMatch) {
  //         month = monthMap[altMatch[1].toUpperCase()] || 0;
  //         year = +altMatch[2];
  //       }
  //     }

  //     return new Date(year, month, day);
  //   }
  // }

  // getflavours() {
  //   this.clusterService.getppuEnabledFlavors(
  //     this.zoneId,
  //     false,
  //     null,
  //     (response) => {
  //       // console.log("response flavours list", response);
  //       this.flavours = response.data.flavor;
  //       console.log("flavours--------------------------------", this.flavours);
  //     },
  //     (error) => {
  //       console.log("error", error);
  //     }
  //   );
  // }
  getaddons() {
    // For customer login, hardcode clusterType to "APP"
    if (this.flags.isCustomer && !this.clusterType) {
      this.clusterType = "APP";
      // console.log("Customer login detected in getaddons, setting clusterType to APP");
    }

    // Increment request ID to invalidate previous requests
    this.currentAddonsRequestId++;
    const requestId = this.currentAddonsRequestId;
    // console.log(`Starting addons request with ID: ${requestId}`);

    this.clusterService.getMSServiceList(
      this.zoneId,
      this.kubernetesVersion,
      this.clusterType,
      (response) => {
        // Check if this request is still the current one
        if (requestId !== this.currentAddonsRequestId) {
          // console.log(`Addons request ${requestId} was superseded by newer request, ignoring response`);
          return;
        }
 
        this.addons = response.data || [];
        // console.log(`Addons request ${requestId} completed successfully:`, this.addons);
        
        // Filter observability tools based on available addons
        this.filterObservabilityTools(this.addons);

      },
      (error) => {
        // Check if this request is still the current one
        if (requestId !== this.currentAddonsRequestId) {
          console.log(`Addons request ${requestId} was superseded by newer request, ignoring error`);
          return;
        }
        console.error(`Error fetching add ons for request ${requestId}:`, error);
      }
    );
  }

  getnetworklist(endpointId?: number) {
    // Use endpointId instead of zoneId for network list
    const networkEndpointId = endpointId || this.zoneId;
    
    // console.log("Calling getNetworkList with endpointId:", networkEndpointId);
    // console.log("Kubernetes Version:", this.kubernetesVersion);
    // console.log("Cluster Type:", this.clusterType);

    // Increment request ID to invalidate previous requests
    this.currentNetworkRequestId++;
    const requestId = this.currentNetworkRequestId;
    // console.log(`Starting network request with ID: ${requestId}`);

    
    /*
    const mockResponse = {
      status: 'success',
      data: {
        data: [
          "calico-v3.25.1",
          "cilium-ebpf-v1.16.4", 
          "cilium-iptables-v1.16.4"
        ],
        status: 'success'
      },
      message: 'success',
      responseCode: 0
    };

    // Use mock response instead of API call
    this.networks = mockResponse.data || [];
    console.log("networks (mock)", this.networks);
    this.populateCNIDriverOptions(mockResponse.data?.data || []);
    return;
    */

    this.clusterService.getNetworkList(
      networkEndpointId,
      this.kubernetesVersion,
      this.clusterType,
      (response) => {
        // Check if this request is still the current one
        if (requestId !== this.currentNetworkRequestId) {
          console.log(`Network request ${requestId} was superseded by newer request, ignoring response`);
          return;
        }
 
        this.networks = response.data || [];
        console.log(`Network request ${requestId} completed successfully:`, this.networks);
        
        // Populate CNI driver options from the response
        this.populateCNIDriverOptions(response.data?.data || []);

      },
      (error) => {
        // Check if this request is still the current one
        if (requestId !== this.currentNetworkRequestId) {
          console.log(`Network request ${requestId} was superseded by newer request, ignoring error`);
          return;
        }
        console.error(`Error fetching networks for request ${requestId}:`, error);
      }
    );
  }


  getostype$(): Observable<any[]> {
    return new Observable(observer => {
      this.clusterService.getppuEnabledImages(
        this.zoneId,
        null,
        false,
        (response) => {
          // Store the response globally for other uses
          this.globalOstypeResponse = response;
          // console.log("🌐 Global OS type response stored:", this.globalOstypeResponse);
          
          this.ostype = response.data.image.options || [];
          // console.log("✅ All fetched flavors (ostype):", this.ostype);
          // console.log("🔍 Kubernetes Version:", this.kubernetesVersion);
          if (!this.kubernetesVersion) {
            console.warn("⚠️ Kubernetes version not available.");
          }
  
          // Filter by Kubernetes version first
          const filteredImages = this.ostype.filter(img => img.label?.includes(this.kubernetesVersion));
          
          // Group by osMake + osVersion combination
          const groupedByMakeAndVersion = filteredImages.reduce((acc, img) => {
            const key = `${img.osMake} ${img.osVersion}`;
            if (!acc[key]) acc[key] = [];
            acc[key].push(img);
            return acc;
          }, {} as Record<string, any[]>);
  
          // console.log("📦 Grouped by osMake + osVersion (filtered by k8s version):", groupedByMakeAndVersion);
  
          const osOptionsMap = new Map<string, any>();
  
          for (const key in groupedByMakeAndVersion) {
            const images = groupedByMakeAndVersion[key];
            // Sort by date to get the latest image for this OS+Version combination
            images.sort((a, b) => this.extractDateFromLabel(b.label).getTime() - this.extractDateFromLabel(a.label).getTime());
            const latestImage = images[0];
            if (latestImage) {
              osOptionsMap.set(key, {
                id: key,
                itemName: key, // This will be "Ubuntu 22.04 LTS" or "Ubuntu 24.04 LTS"
                osId: latestImage.id,
                osModel: latestImage.osModel,
                osMake: latestImage.osMake,
                osVersion: latestImage.osVersion,
                hypervisor: latestImage.hypervisor,
                // Store all images for this combination for later filtering
                allImages: images
              });
            }
          }
  
          this.osOptions = Array.from(osOptionsMap.values());
          // console.log("✅ Final OS options:", this.osOptions);
  
          const workerNodeStep = this.stepDefinitions.find(step => step.label.includes('Worker Node Pool'));
          if (workerNodeStep) {
            const osControl = workerNodeStep.formControls.find(ctrl => ctrl.name === 'operatingSystem');
            if (osControl) {
              osControl.options = this.osOptions;
            }
          }
  
          observer.next(this.ostype);
          observer.complete();
        },
        (error) => {
          console.error('❌ getostype error:', error);
          observer.error(error);
        }
      );
    });
  }

  private extractDateFromLabel(label: string): Date {
    const monthMap = {
      JAN: 0, FEB: 1, MAR: 2, APR: 3, MAY: 4, JUN: 5,
      JUL: 6, AUG: 7, SEP: 8, OCT: 9, NOV: 10, DEC: 11
    };
    const dateRegex = /(\d{2})([A-Z]{3})(\d{4})/i;
    const altRegex = /([A-Z]{3})(\d{4})/i;

    let day = 1, month = 0, year = 1970;

    const match = label.match(dateRegex);
    if (match) {
      day = +match[1];
      month = monthMap[match[2].toUpperCase()] || 0;
      year = +match[3];
    } else {
      const altMatch = label.match(altRegex);
      if (altMatch) {
        month = monthMap[altMatch[1].toUpperCase()] || 0;
        year = +altMatch[2];
      }
    }

    return new Date(year, month, day);
  }
  

  getflavours$(): Observable<any[]> {
    return new Observable(observer => {
      this.clusterService.getppuEnabledFlavors(
        this.zoneId,
        false,
        null,
        (response) => {
          this.flavours = response.data.flavor || [];
          observer.next(this.flavours);
          observer.complete();
        },
        (error) => {
          console.error("getflavours error:", error);
          observer.error(error);
        }
      );
    });
  }

  updateNetworkSetupDropdowns() {
    const networkSetupStep = this.stepDefinitions.find(step => step.label === 'Network Setup');
    if (networkSetupStep) {
      const businessUnitControl = networkSetupStep.formControls.find(ctrl => ctrl.name === 'businessUnit');
      const environmentControl = networkSetupStep.formControls.find(ctrl => ctrl.name === 'environment');
      const zoneControl = networkSetupStep.formControls.find(ctrl => ctrl.name === 'zone');

      if (businessUnitControl) {
        businessUnitControl.options = this.businessUnitOptions;
      }

      // Clear environment and zone options when business units change
      if (environmentControl) {
        environmentControl.options = [];
        environmentControl.value = null;
      }

      if (zoneControl) {
        zoneControl.options = [];
        zoneControl.value = null;
      }
    }
  }

  clearNetworkSetupDropdowns() {
    const networkSetupStep = this.stepDefinitions.find(step => step.label === 'Network Setup');
    if (networkSetupStep) {
      const businessUnitControl = networkSetupStep.formControls.find(ctrl => ctrl.name === 'businessUnit');
      const environmentControl = networkSetupStep.formControls.find(ctrl => ctrl.name === 'environment');
      const zoneControl = networkSetupStep.formControls.find(ctrl => ctrl.name === 'zone');

      if (businessUnitControl) {
        businessUnitControl.options = [];
        businessUnitControl.value = null;
      }

      if (environmentControl) {
        environmentControl.options = [];
        environmentControl.value = null;
      }

      if (zoneControl) {
        zoneControl.options = [];
        zoneControl.value = null;
      }
    }
  }



  handleDropdownChange(event: { controlName: string, value: any, itemName: string }) {

    switch (event.controlName) {
      case 'engagements':
        this.onEngagementChange(event);
        break;
      case 'businessUnit':
        // console.log("businessUnit", event, event.value);
        // Extract ID from the object
        const businessUnitId = event.value?.id || event.value;
        this.onBusinessUnitChange(businessUnitId);
        break;
      case 'environment':
        // console.log("onEnvironmentChange", event, event.value);
        // Extract itemName from the object
        const environmentName = event.value?.itemName || event.value;
        this.onEnvironmentChange(environmentName);
        break;
      case 'zone':
        this.onZoneChange(event);
        break;
      case 'operatingSystem':
        // Extract itemName from the object (this will be "Ubuntu 22.04 LTS" format)
        const selectedOSVersion = event.value?.itemName || event.value;
        this.selectedOSVersion = selectedOSVersion; // Store for delayed processing
        if (this.flavours?.length && this.ostype?.length) {
          this.onOSTypeSelect(selectedOSVersion); // If data already loaded
        }
        break;
      case 'node type':
        // Extract itemName from the object
        this.selectedNodeType = event.value?.itemName || event.value;
        
        // Store both display and original values for API calls
        if (event.value?.originalValue) {
          this.selectedNodeTypeOriginal = event.value.originalValue;
        } else {
          // Fallback: find the original value from options
          const option = this.nodeTypeOptions.find(opt => opt.itemName === this.selectedNodeType);
          this.selectedNodeTypeOriginal = option?.originalValue || this.selectedNodeType;
        }
        
        this.onNodeTypeSelect(this.selectedNodeType);
        break;
    }
  }


  // onOSTypeSelect(selectedOS: string) {
  //   // console.log("Selected OS:", selectedOS);

  //   // Find the step once and reuse
  //   const workerNodeStep = this.stepDefinitions.find(step => step.label.includes('Worker Node Pool'));
  //   if (!workerNodeStep) return;

  //   // --- VERSION LOGIC ---
  //   const filteredImages = this.ostype.filter(img => img.osMake === selectedOS);
  //   const uniqueVersions = [...new Set(filteredImages.map(img => img.osVersion))];

  //   this.versionOptions = uniqueVersions.map((ver, index) => ({
  //     id: index + 1,
  //     itemName: ver
  //   }));
  //   // console.log("Version Options", this.versionOptions);

  //   const versionControl = workerNodeStep.formControls.find(ctrl => ctrl.name === 'version');
  //   if (versionControl) {
  //     versionControl.options = this.versionOptions;
  //   }

  //   // --- FLAVOUR LOGIC ---
  //   let osMake = '';
  //   if (selectedOS === 'Ubuntu') osMake = 'ubuntu';
  //   else if (selectedOS === 'Red Hat Enterprise Linux') osMake = 'rhel';
  //   else if (selectedOS === 'CentOS') osMake = 'centos';

  //   if (this.flavours.length > 0) {
  //     const matchedFlavors = this.flavours?.filter(
  //       flavor => flavor.applicationType === 'Container' && flavor.osModel === osMake
  //     );

  //     // Remove duplicates by flavor.label
  //     const uniqueLabels = new Set();
  //     const formattedFlavors = [];

  //     matchedFlavors.forEach((flavor) => {
  //       if (!uniqueLabels.has(flavor.label)) {
  //         uniqueLabels.add(flavor.label);

  //         formattedFlavors.push({
  //           id: flavor.artifactId,
  //           itemName: `${Math.round(flavor.vRam / 1024)} GB RAM / ${flavor.vCpu} vCPU / ${flavor.vDisk} GB Storage`,
  //           flavordisk: flavor.vDisk,
  //           flavorskucode: flavor.skuCode,
  //           flavorname: flavor.FlavorName
  //         });
  //       }
  //     });

  //     this.flavourOptions = formattedFlavors;
  //     console.log("Flavour Options", this.flavourOptions);

  //     const flavourControl = workerNodeStep.formControls.find(ctrl => ctrl.name === 'flavour');

  //     if (flavourControl) {
  //       flavourControl.options = this.flavourOptions;
  //       flavourControl._originalOptions = [...this.flavourOptions];
  //     }
  //   }
  // }

  onOSTypeSelect(selectedOSVersion: string) {
    if (!this.flavours?.length || !this.ostype?.length) {
      console.warn('Skipping OS select: data not ready.');
      return;
    }

    const workerNodeStep = this.stepDefinitions.find(step => step.label.includes('Worker Node Pool'));
    if (!workerNodeStep) return;

    // Find the selected OS option to get the osMake and osVersion
    const selectedOSOption = this.osOptions.find(option => option.itemName === selectedOSVersion);
    if (!selectedOSOption) {
      console.warn('Selected OS option not found:', selectedOSVersion);
      return;
    }

    const { osMake, osVersion, allImages } = selectedOSOption;

    // Apply date filtering to get the latest image for this OS+Version combination
    const latestImage = allImages.sort((a, b) => 
      this.extractDateFromLabel(b.label).getTime() - this.extractDateFromLabel(a.label).getTime()
    )[0];

    // Store the selected OS details for later use
    this.selectedOS = {
      osMake: latestImage.osMake,
      osVersion: latestImage.osVersion,
      osModel: latestImage.osModel,
      osId: latestImage.id,
      hypervisor: latestImage.hypervisor
    };

    // --- FLAVOUR LOGIC ---
    const osMakeMap = {
      'Ubuntu': 'ubuntu',
      'Red Hat Enterprise Linux': 'rhel',
      'CentOS': 'centos'
    };

    const osMakeForFlavor = osMakeMap[osMake] || '';

    const matchedFlavors = this.flavours?.filter(
      flavor => flavor.applicationType === 'Container' && flavor.osModel === osMakeForFlavor
    );

    // --- NODE TYPE LOGIC ---
    // Extract unique flavorCategory values for Node Type dropdown
    const uniqueFlavorCategories = [...new Set(matchedFlavors.map(flavor => flavor.flavorCategory))];
    
    // Store original flavor categories for API calls
    this.originalFlavorCategories = uniqueFlavorCategories;
    
    // Create display options with direct mapping for user-friendly names
    // This allows users to see "General Purpose" instead of "generalPurpose"
    // while preserving the original value "generalPurpose" for API calls
    this.nodeTypeOptions = uniqueFlavorCategories.map((category, index) => ({
      id: index + 1,
      itemName: this.getNodeTypeDisplayName(category),
      originalValue: category // Store original value for API calls
    }));

    // Populate Node Type dropdown
    const nodeTypeControl = workerNodeStep.formControls.find(ctrl => ctrl.name === 'node type');
    if (nodeTypeControl) {
      nodeTypeControl.options = this.nodeTypeOptions;
      nodeTypeControl._originalOptions = [...this.nodeTypeOptions]; // Store original options
      // Add a custom property to store both display and original values
      nodeTypeControl.storeOriginalValue = true;
      // Clear the selected value when OS changes
      nodeTypeControl.value = null;
      this.selectedNodeType = '';
      this.selectedNodeTypeOriginal = '';
    }

    // For master flavors, populate directly without Node Type filtering
    const uniqueLabels = new Set();
    const formattedFlavors = [];

    matchedFlavors.forEach((flavor) => {
      if (!uniqueLabels.has(flavor.label)) {
        uniqueLabels.add(flavor.label);
        formattedFlavors.push({
          artifactId: flavor.artifactId,
          id: `${flavor.vCpu} vCPU / ${Math.round(flavor.vRam / 1024)} GB RAM / ${flavor.vDisk} GB Storage`,
          itemName: `${flavor.vCpu} vCPU / ${Math.round(flavor.vRam / 1024)} GB RAM / ${flavor.vDisk} GB Storage`,
          flavordisk: flavor.vDisk,
          flavorskucode: flavor.skuCode,
          flavorname: flavor.FlavorName
        });
      }
    });

    // Sort master flavors by vCPU in ascending order
    formattedFlavors.sort((a, b) => {
      const aVcpu = parseInt(a.id.match(/(\d+)\s*vCPU/)?.[1] || '0');
      const bVcpu = parseInt(b.id.match(/(\d+)\s*vCPU/)?.[1] || '0');
      return aVcpu - bVcpu;
    });

    // Store all matched flavors for later filtering
    this.flavourOptions = formattedFlavors;

    // Populate master flavor options directly (no Node Type filtering)
    const masterFlavourControl = workerNodeStep.formControls.find(ctrl => ctrl.name === 'masterFlavor');
    if (masterFlavourControl) {
      masterFlavourControl.options = this.flavourOptions;
      masterFlavourControl._originalOptions = [...this.flavourOptions];
    }

    // Clear worker node flavor options until Node Type is selected
    const flavourControl = workerNodeStep.formControls.find(ctrl => ctrl.name === 'flavour');
    if (flavourControl) {
      flavourControl.options = [];
      flavourControl._originalOptions = [...this.flavourOptions]; // Store original options for filtering
    }
  }

  onNodeTypeSelect(selectedNodeType: string) {
    if (!this.flavours?.length || !this.selectedOS) {
      console.warn('Skipping Node Type select: data not ready.');
      return;
    }

    const workerNodeStep = this.stepDefinitions.find(step => step.label.includes('Worker Node Pool'));
    if (!workerNodeStep) return;

    // Get the OS mapping - now selectedOS is an object with osMake property
    const osMakeMap = {
      'Ubuntu': 'ubuntu',
      'Red Hat Enterprise Linux': 'rhel',
      'CentOS': 'centos'
    };
    const osMake = osMakeMap[this.selectedOS.osMake] || '';

    // Find the original flavor category value from the selected node type option
    // This ensures we use the original API value (e.g., "generalPurpose") for filtering
    // instead of the display value (e.g., "General Purpose")
    const selectedNodeTypeOption = this.nodeTypeOptions.find(option => 
      option.itemName === selectedNodeType || option.originalValue === selectedNodeType
    );
    const originalFlavorCategory = selectedNodeTypeOption?.originalValue || selectedNodeType;

    // Filter flavors based on OS, application type, and selected node type (use original value)
    // This ensures proper API compatibility while maintaining user-friendly display
    const matchedFlavors = this.flavours?.filter(
      flavor => flavor.applicationType === 'Container' && 
                flavor.osModel === osMake && 
                flavor.flavorCategory === originalFlavorCategory
    );

    const uniqueLabels = new Set();
    const formattedFlavors = [];

    matchedFlavors.forEach((flavor) => {
      if (!uniqueLabels.has(flavor.label)) {
        uniqueLabels.add(flavor.label);
        formattedFlavors.push({
          artifactId: flavor.artifactId,
          id: `${flavor.vCpu} vCPU / ${Math.round(flavor.vRam / 1024)} GB RAM / ${flavor.vDisk} GB Storage`,
          itemName: `${flavor.vCpu} vCPU / ${Math.round(flavor.vRam / 1024)} GB RAM / ${flavor.vDisk} GB Storage`,
          flavordisk: flavor.vDisk,
          flavorskucode: flavor.skuCode,
          flavorname: flavor.FlavorName
        });
      }
    });

    // Sort worker node flavors by vCPU in ascending order
    formattedFlavors.sort((a, b) => {
      const aVcpu = parseInt(a.id.match(/(\d+)\s*vCPU/)?.[1] || '0');
      const bVcpu = parseInt(b.id.match(/(\d+)\s*vCPU/)?.[1] || '0');
      return aVcpu - bVcpu;
    });

    // Update worker node flavor options
    const flavourControl = workerNodeStep.formControls.find(ctrl => ctrl.name === 'flavour');
    if (flavourControl) {
      flavourControl.options = formattedFlavors;
    }
  }

  getclusterTemplate() {

    this.clusterService.getClusterDetails(1,
        (response) => {
           console.log("response for cluster template", response)
           this.stepDefinitions= response.data
// this.stepDefinitions = [
//   {
//       "label": "Cluster Configuration",
//       "optional": false,
//       "description": "Configure and manage cluster settings to enhance performance of the application.",
//       "reviewLabel": "Cluster Configuration",
//       "formControls": [
//           {
//               "name": "engagements",
//               "type": "singleSelect",
//               "label": "Engagements",
//               "options": [],
//               "description": "Select the engagement under which the cluster is to be deployed.",
//               "validations": {
//                   "message": "Engagement is required.",
//                   "required": true
//               }
//           },
//           {
//               "name": "clusterName",
//               "type": "text",
//               "label": "Cluster Name",
//               "description": "Assign a unique name to your cluster for easy identification",
//               "validations": {
//                   "message": "Cluster Name should start with a letter and can have only 3 to 18 characters (numbers,alphabets,hyphens and underscores).",
//                   "pattern": "^[a-zA-Z][a-zA-Z0-9-]{2,17}$",
//                   "required": true
//               },
//               "customValidation": true
//           },
//           {
//               "name": "datacenter",
//               "type": "singleSelect",
//               "label": "Data Center",
//               "options": [],
//               "description": "Select the location where you want your cluster to get deployed .",
//               "validations": {
//                   "message": "Data Center is required.",
//                   "required": true
//               }
//           },
//           {
//               "name": "kubernetesVersion",
//               "type": "singleSelect",
//               "label": "Kubernetes version",
//               "options": [],
//               "learnmore": "https://kubernetes.io/releases/",
//               "description": "Pick a Kubernetes version.",
//               "validations": {
//                   "message": "Kubernetes is required.",
//                   "required": true
//               }
//           },
//           {
//               "name": "clusterType",
//               "type": "radio",
//               "label": "Cluster Type",
//               "options": [
//                   {
//                       "id": "APP",
//                       "title": "Application Cluster",
//                       "activeIcon": "./assets/images/paas/application-cluster-active.svg",
//                       "defaultIcon": "./assets/images/paas/application-cluster-default.svg"
//                   },
//                   {
//                       "id": "MGMT",
//                       "title": "Management Cluster",
//                       "activeIcon": "./assets/images/paas/managed-cluster-active.svg",
//                       "defaultIcon": "./assets/images/paas/managed-cluster-default.svg"
//                   }
//               ],
//               "description": "Balanced compute,memory and storage for a wide range of workloads",
//               "validations": {
//                   "message": "Cluster type is required.",
//                   "required": true
//               }
//           },
//           {
//               "info": "",
//               "name": "controlPlaneType",
//               "type": "radio",
//               "label": "Control Plane Type",
//               "options": [
//                   {
//                       "id": "dedicated",
//                       "title": "Dedicated Control Plane",
//                       "subtitle": "Deploy dedicated master nodes for your cluster.",
//                       "activeIcon": "./assets/images/paas/dedicated-control-plane-active.svg",
//                       "defaultIcon": "./assets/images/paas/dedicated-control-plane-default.svg"
//                   },
//                   {
//                       "id": "virtual",
//                       "title": "Virtual Control Plane",
//                       "subtitle": "Deploy your cluster under a shared control plane.",
//                       "activeIcon": "./assets/images/paas/virtual-control-plane-active.svg",
//                       "defaultIcon": "./assets/images/paas/virtual-control-plane-default.svg"
//                   }
//               ],
//               "description": "Choose the most suitable option, Standard or High Availability according to your requirements",
//               "validations": {
//                   "message": "Control Plane type is required.",
//                   "required": true
//               },
//               "dependentControl": {
//                   "controlName": "dedicatedControlPlaneType",
//                   "triggerValue": "dedicated",
//                   "externalControlNames": "masterFlavor"
//               }
//           },
//           {
//               "name": "dedicatedControlPlaneType",
//               "type": "singleSelect",
//               "label": "Choose Dedicated Control Plane Type",
//               "options": [
//                   {
//                       "id": "Single Master",
//                       "itemName": "Standard",
//                       "description": "Provide 1 master node redundancy"
//                   },
//                   {
//                       "id": "High availability",
//                       "itemName": "High Availability",
//                       "description": "Provide 3 master node redundancy"
//                   }
//               ],
//               "dependency": [
//                   {
//                       "field": "controlPlaneType",
//                       "value": "dedicated"
//                   }
//               ],
//               "validations": {
//                   "message": "Control Plane Type is required."
//               }
//           },
//           {
//               "name": "cnidriver",
//               "type": "singleSelect",
//               "label": "Choose a CNI Driver",
//               "options": [],
//               "description": "Select the container network interface (CNI) plugin that will manage networking for your pods."
//           },
//           {
//               "name": "Tags",
//               "type": "tags",
//               "label": "Tags",
//               "options": [],
//               "learnmore": "https://ipcloud.tatacommunications.com/docs/docs/user-docs/iks/core_concepts#associate-tags",
//               "description": "Add key-value pairs to define custom labels or settings.",
//               "validations": null
//           }
//       ],
//       "dottedStepperIcon": "./assets/images/paas/dotted-cluster-configuration.svg",
//       "defaultStepperIcon": "./assets/images/paas/default-cluster-configuration.svg",
//       "completedStepperIcon": "./assets/images/paas/complete.svg"
//   },
//   {
//       "label": "Network Setup",
//       "optional": false,
//       "description": "Configure essential settings for your network.",
//       "reviewLabel": "Networking",
//       "formControls": [
//           {
//               "name": "businessUnit",
//               "type": "singleSelect",
//               "label": "Business Unit",
//               "options": [],
//               "description": "Select the department under which the cluster is to be deployed.",
//               "validations": {
//                   "message": "Business Unit is required.",
//                   "required": true
//               },
//               "addNewOption": true
//           },
//           {
//               "url": "https://ipcloud.tatacommunications.com/cloud/console/ipc/#/bu/list",
//               "name": "createbusinessUnit",
//               "type": "createbutton",
//               "label": "Create Business Unit"
//           },
//           {
//               "name": "environment",
//               "type": "singleSelect",
//               "label": "Environment",
//               "options": [],
//               "learnmore": "https://ipcloud.tatacommunications.com/docs/docs/environments/",
//               "description": "Choose the deployment stage.",
//               "validations": {
//                   "message": "Environment is required.",
//                   "required": true
//               },
//               "addNewOption": true
//           },
//           {
//               "url": "https://ipcloud.tatacommunications.com/cloud/console/ipc/#/env/list",
//               "name": "createenvironment",
//               "type": "createbutton",
//               "label": "Create Environment"
//           },
//           {
//               "name": "zone",
//               "type": "singleSelect",
//               "label": "Zone",
//               "options": [],
//               "learnmore": "https://ipcloud.tatacommunications.com/docs/docs/zones",
//               "description": "Specify the logical network(VLAN) for cluster deployment.",
//               "validations": {
//                   "message": "Zone is required.",
//                   "required": true
//               },
//               "addNewOption": true
//           },
//           {
//               "url": "https://ipcloud.tatacommunications.com/cloud/console/ipc/#/zone/page/list",
//               "name": "createzone",
//               "type": "createbutton",
//               "label": "Create Zone"
//           }
//       ],
//       "dottedStepperIcon": "./assets/images/paas/dotted-Network-Setup.svg",
//       "defaultStepperIcon": "./assets/images/paas/default-Network-Setup.svg",
//       "completedStepperIcon": "./assets/images/paas/complete.svg"
//   },
//   {
//       "type": "table",
//       "label": "Configuring Worker Node Pool",
//       "optional": false,
//       "tableTitle": "List of Worker Node Pool",
//       "description": "Scale your Cluster with Worker Nodes",
//       "reviewLabel": "Worker Node Pool",
//       "formControls": [
//           {
//               "icon": "./assets/images/paas/os-icon.svg",
//               "name": "operatingSystem",
//               "type": "singleSelect",
//               "label": "Operating System",
//               "options": [],
//               "cardPopup": true,
//               "learnmore": "https://kubernetes.io/docs/concepts/windows/",
//               "buttonLabel": "Change Operating Environment",
//               "description": "You are currently using an OS version configured for your worker nodes.",
//               "imageMapping": {
//                   "SUSE": "./assets/images/os_icons/suse.svg",
//                   "CentOS": "./assets/images/os_icons/centos.svg",
//                   "Ubuntu": "./assets/images/os_icons/ubuntu.svg",
//                   "Windows": "./assets/images/os_icons/windows.svg",
//                   "Oracle Linux": "./assets/images/os_icons/oel.svg",
//                   "Red Hat Enterprise Linux": "./assets/images/os_icons/redhat.svg"
//               },
//               "btnDescription": "You are currently using an OS version configured for your worker nodes.",
//               "popupDescription": "Pick the OS version to switch. The system will update after approval."
//           },
//           {
//               "icon": "./assets/images/paas/os-icon.svg",
//               "name": "masterFlavor",
//               "type": "singleSelect",
//               "label": "Master Flavor",
//               "options": [],
//               "cardPopup": true,
//               "buttonLabel": "Change Master Flavor",
//               "description": "Select master node compute configuration.",
//               "validations": {
//                   "message": "Master Flavor is required.",
//                   "required": true
//               },
//               "subdescription": "You are currently using an OS version configured for your worker nodes.",
//               "externalDependency": {
//                   "disable": false,
//                   "dependency": [
//                       {
//                           "field": "controlPlaneType",
//                           "value": "dedicated"
//                       }
//                   ],
//                   "dependentControlNames": [
//                       "controlPlaneType"
//                   ]
//               }
//           },
//           {
//               "name": "workerNodePoolName",
//               "type": "text",
//               "label": "Worker Node Pool Name",
//               "description": "Assign a name to worker pool node.",
//               "tableColumn": true,
//               "validations": {
//                   "message": "Requires only 5 lowercase alphanumeric characters",
//                   "pattern": "^[a-z0-9]{1,5}$",
//                   "required": true
//               }
//           },
//           {
//               "name": "node type",
//               "type": "singleSelect",
//               "label": "Node Type",
//               "options": [],
//               "tableColumn": true,
//               "validations": {
//                   "message": "Node Type is required.",
//                   "required": true
//               }
//           },
//           {
//               "name": "flavour",
//               "type": "singleSelect",
//               "label": "Flavour",
//               "options": [],
//               "learnmore": "https://ipcloud.tatacommunications.com/docs/docs/user-docs/iks/core_concepts#worker-nodes",
//               "description": "Select worker pool node compute configuration.",
//               "tableColumn": true,
//               "validations": {
//                   "message": "Flavour is required.",
//                   "required": true
//               }
//           },
//           {
//               "name": "deploymentScaling",
//               "type": "number",
//               "label": "Fixed Replica Count",
//               "inputType": "number",
//               "learnmore": "https://ipcloud.tatacommunications.com/docs/docs/user-docs/iks/core_concepts#worker-nodes",
//               "description": "Keep your app fast and available by adjusting the scale.",
//               "tableColumn": true,
//               "validations": {
//                   "max": 8,
//                   "min": 1,
//                   "message": "",
//                   "required": true
//               }
//           },
//           {
//               "name": "enable Replica",
//               "type": "checkbox",
//               "label": "Enable if you want to auto scale on compute requirements.",
//               "description": "Keep your app fast and available by adjusting the scale.",
//               "tableColumn": false
//           },
//           {
//               "name": "scaledReplica",
//               "type": "number",
//               "label": "Scaled Replica Count",
//               "disabled": false,
//               "inputType": "number",
//               "dependency": [
//                   "enable Replica"
//               ],
//               "reviewLabel": "Max Replica Count",
//               "tableColumn": true,
//               "validations": {
//                   "max": 8,
//                   "min": 1,
//                   "message": ""
//               }
//           },
//           {
//               "name": "addWorkerNodePool",
//               "type": "button",
//               "label": "Add Worker Node Pool",
//               "style": "primary",
//               "disabled": false,
//               "tableLabel": "List of Worker Node Pools",
//               "description": "Stay connected with the newest additions to our node list, ensuring seamless operations.",
//               "discardButton": true
//           }
//       ],
//       "toasterDelete": {
//           "type": "delete",
//           "message": "Worker Node Pool Deleted",
//           "subMessage": "The worker node pool configuration has been deleted"
//       },
//       "toasterSuccess": {
//           "type": "success",
//           "message": "Worker Node Pool Updated",
//           "subMessage": "The worker node pool configuration has been updated"
//       },
//       "tableDescription": "Stay connected with the newest additions to our node list, ensuring seamless operations.",
//       "dottedStepperIcon": "./assets/images/paas/dotted-configure-worker-node-pool.svg",
//       "defaultStepperIcon": "./assets/images/paas/default-configure-worker-node-pool.svg",
//       "completedStepperIcon": "./assets/images/paas/complete.svg",
//       "showFormFieldsOnEdit": true,
//       "hideFormFieldsAfterFirstAdd": true
//   },
//   {
//       "type": "table",
//       "label": "Storage Class Configuration",
//       "optional": true,
//       "tableTitle": "Configured Storage Classes",
//       "description": "This section enables efficient storage configuration for persistent volumes.",
//       "reviewLabel": "Persistent Volume Claim",
//       "formControls": [
//           {
//               "name": "persistentVolumeName",
//               "type": "text",
//               "label": "Persistent Volume Name",
//               "description": "Provide a name for the storage volume.",
//               "tableColumn": true,
//               "validations": {
//                   "message": "Persistent Volume Name is required.",
//                   "required": true
//               }
//           },
//           {
//               "name": "classConfiguration",
//               "type": "singleSelect",
//               "label": "Class Configuration",
//               "options": [
//                   {
//                       "id": "IOPS 1",
//                       "itemName": "IOPS 1"
//                   },
//                   {
//                       "id": "IOPS 5",
//                       "itemName": "IOPS 5"
//                   }
//               ],
//               "learnmore": "https://ipcloud.tatacommunications.com/docs/docs/user-docs/iks/core_concepts#persistentvolume-pv-and-persistentvolumeclaim-pvc",
//               "description": "Choose the appropriate IOPS version based on your performance requirements.",
//               "tableColumn": true,
//               "validations": {
//                   "message": "Class Configuration is required.",
//                   "required": true
//               },
//               "isSelectedValue": [],
//               "removeSelectedValue": true
//           },
//           {
//               "name": "capacity",
//               "type": "number",
//               "label": "Capacity (GB)",
//               "learnmore": "https://ipcloud.tatacommunications.com/docs/docs/user-docs/iks/core_concepts#persistentvolume-pv-and-persistentvolumeclaim-pvc",
//               "description": "Defines the storage capacity in GB, up to a maximum of 7000 GB.",
//               "tableColumn": true,
//               "validations": {
//                   "min": 1,
//                   "message": "Capacity is required.",
//                   "required": true
//               }
//           },
//           {
//               "name": "addpvc",
//               "type": "button",
//               "label": "Add PVC",
//               "style": "primary",
//               "disabled": false,
//               "tableLabel": "Configuring PVCs for Cluster Storage",
//               "description": "It provides a feature to configuring Persistent Volume Claims (PVCs) to bind to Persistent Volumes (PVs) later.",
//               "discardButton": true
//           }
//       ],
//       "toasterDelete": {
//           "type": "delete",
//           "message": "Persistent Volume Claim Deleted",
//           "subMessage": "The Storage Class Configuration has been deleted"
//       },
//       "toasterSuccess": {
//           "type": "success",
//           "message": "Storage Class Configuration Updated",
//           "subMessage": "The storage class configuration has been updated"
//       },
//       "tableDescription": "These storage classes define how persistent volumes are provisioned across your cluster.",
//       "dottedStepperIcon": "./assets/images/paas/dotted-storage-class-configuration.svg",
//       "defaultStepperIcon": "./assets/images/paas/default-storage-class-configuration.svg",
//       "completedStepperIcon": "./assets/images/paas/complete.svg",
//       "showFormFieldsOnEdit": true,
//       "hideFormFieldsAfterFirstAdd": true
//   },
//   {
//       "label": "Backup Management",
//       "optional": true,
//       "description": "Take control of your data's security with effective backup management for your cluster.",
//       "reviewLabel": "Backup",
//       "formControls": [
//           {
//               "info": "",
//               "name": "backupType",
//               "type": "backupType",
//               "label": "Backup Type",
//               "description": "Choose whether to backup all cluster data or backup specific namespaces only.",
//               "validations": {
//                   "message": "Backup type is required.",
//                   "required": true
//               }
//           },
//           {
//               "name": "byLabels",
//               "type": "filterbylabel",
//               "label": "Filter By Labels",
//               "options": [],
//               "description": "Label filters help narrow down restore jobs across large environments.",
//               "validations": {
//                   "message": "required.",
//                   "required": true
//               }
//           },
//           {
//               "name": "backupNonNamespacedResources",
//               "type": "backupNonNamespacedResourcescheckbox",
//               "label": "Backup non-namespaced resources",
//               "description": "Backup non-namespaced resources as well"
//           },
//           {
//               "name": "storagePool",
//               "type": "storage",
//               "label": "Storage Pool Configuration",
//               "description": "Configure storage settings for backup operations.",
//               "validations": {
//                   "message": "Storage configuration is required.",
//                   "required": true
//               }
//           },
//           {
//               "name": "fetcapacity",
//               "type": "number",
//               "label": "FET Reserve Capacity",
//               "inputType": "number",
//               "learnmore": "",
//               "description": "Specify the total backup storage size to allocate based on your expected backup data and retention needs.",
//               "validations": {
//                   "max": 10000,
//                   "min": 1,
//                   "message": "FET reserve capacity is required.",
//                   "required": true
//               }
//           },
//           {
//               "name": "backupFrequency",
//               "type": "multiSelect",
//               "label": "Frequency",
//               "options": [
//                   {
//                       "id": "1",
//                       "itemName": "Daily"
//                   },
//                   {
//                       "id": "2",
//                       "itemName": "Weekly"
//                   },
//                   {
//                       "id": "3",
//                       "itemName": "Monthly"
//                   },
//                   {
//                       "id": "4",
//                       "itemName": "Yearly"
//                   }
//               ],
//               "description": "Choose the backup frequency from daily, weekly, monthly, or yearly options."
//           },
//           {
//               "name": "Daily",
//               "dependancy": "backupFrequency",
//               "formControls": [
//                   {
//                       "name": "backupTime",
//                       "type": "time",
//                       "label": "Backup Time",
//                       "value": "22:00",
//                       "description": "Choose the backup frequency from daily, weekly, monthly, or yearly options.",
//                       "placeholder": "00:00:00",
//                       "timezoneLabel": "IST"
//                   },
//                   {
//                       "max": 7,
//                       "min": 1,
//                       "name": "retentionWindow",
//                       "type": "number",
//                       "unit": "Days",
//                       "label": "Retention Window",
//                       "value": "7",
//                       "description": "Set the range between 1 to 7 days",
//                       "placeholder": "Ex: 7"
//                   },
//                   {
//                       "name": "backupType",
//                       "type": "dropdown",
//                       "label": "Backup Type",
//                       "value": "Full",
//                       "options": [
//                           "Full",
//                           "Incremental"
//                       ],
//                       "description": "Select between full or incremental backup.",
//                       "placeholder": "Select"
//                   },
//                   {
//                       "max": 100,
//                       "min": 1,
//                       "name": "alwaysExecutes",
//                       "type": "number",
//                       "label": "Always Executes",
//                       "value": "1",
//                       "description": "Select any number between 1 and 100.",
//                       "placeholder": "Ex: 30"
//                   },
//                   {
//                       "text": "Note: By selecting 'Daily' as the backup frequency, you need to specify the start time, retention period, executive times and the backup type.",
//                       "type": "note"
//                   }
//               ]
//           },
//           {
//               "name": "Weekly",
//               "dependancy": "backupFrequency",
//               "formControls": [
//                   {
//                       "name": "backupTime",
//                       "type": "time",
//                       "label": "Backup Time",
//                       "value": "22:00",
//                       "description": "Specify the time in hours, minutes, and seconds.",
//                       "placeholder": "00:00:00",
//                       "timezoneLabel": "IST"
//                   },
//                   {
//                       "max": 30,
//                       "min": 7,
//                       "name": "retentionWindow",
//                       "type": "number",
//                       "unit": "Days",
//                       "label": "Retention Window",
//                       "value": "30",
//                       "description": "Set the range between 7 to 30 days.",
//                       "placeholder": "Ex: 18"
//                   },
//                   {
//                       "name": "backupDay",
//                       "type": "dropdown",
//                       "label": "Backup Day",
//                       "value": "Sunday",
//                       "options": [
//                           "Sunday",
//                           "Monday",
//                           "Tuesday",
//                           "Wednesday",
//                           "Thursday",
//                           "Friday",
//                           "Saturday"
//                       ],
//                       "description": "Choose any one day of the week.",
//                       "placeholder": "Select"
//                   },
//                   {
//                       "max": 100,
//                       "min": 1,
//                       "name": "alwaysExecutes",
//                       "type": "number",
//                       "label": "Always Executes",
//                       "value": "1",
//                       "description": "Select any number between 1 and 100.",
//                       "placeholder": "Ex: 30"
//                   },
//                   {
//                       "text": "Note: By selecting 'Weekly' as the backup frequency, you need to specify the start time, retention period, executive times and the backup day.",
//                       "type": "note"
//                   }
//               ]
//           },
//           {
//               "name": "Monthly",
//               "dependancy": "backupFrequency",
//               "formControls": [
//                   {
//                       "name": "backupTime",
//                       "type": "time",
//                       "label": "Backup Time",
//                       "value": "22:00",
//                       "description": "Specify the time in hours, minutes, and seconds.",
//                       "placeholder": "00:00:00",
//                       "timezoneLabel": "IST"
//                   },
//                   {
//                       "max": 360,
//                       "min": 28,
//                       "name": "retentionWindow",
//                       "type": "number",
//                       "unit": "Days",
//                       "label": "Retention Window",
//                       "value": "90",
//                       "description": "Set the range between 28 to 360 days.",
//                       "placeholder": "Ex: 45"
//                   },
//                   {
//                       "max": 31,
//                       "min": 1,
//                       "name": "backupDate",
//                       "type": "number",
//                       "label": "Backup Date",
//                       "value": 28,
//                       "description": "Enter any one date of the month.",
//                       "placeholder": "Ex: 30"
//                   },
//                   {
//                       "max": 100,
//                       "min": 1,
//                       "name": "alwaysExecutes",
//                       "type": "number",
//                       "label": "Always Executes",
//                       "value": "1",
//                       "description": "Select any number between 1 and 100.",
//                       "placeholder": "Ex: 30"
//                   },
//                   {
//                       "text": "Note: By selecting 'Monthly' as the backup frequency, you need to specify the start time, retention period, executive times and the backup date.",
//                       "type": "note"
//                   }
//               ]
//           },
//           {
//               "name": "Yearly",
//               "dependancy": "backupFrequency",
//               "formControls": [
//                   {
//                       "name": "backupTime",
//                       "type": "time",
//                       "label": "Backup Time",
//                       "value": "22:00",
//                       "description": "Specify the time in hours, minutes, and seconds.",
//                       "placeholder": "00:00:00",
//                       "timezoneLabel": "IST"
//                   },
//                   {
//                       "max": 1825,
//                       "min": 365,
//                       "name": "retentionWindow",
//                       "type": "number",
//                       "unit": "Days",
//                       "label": "Retention Window",
//                       "value": "730",
//                       "description": "Set the range between 365 to 1825 days",
//                       "placeholder": "Ex: 371"
//                   },
//                   {
//                       "name": "backupMonth",
//                       "type": "dropdown",
//                       "label": "Backup Month",
//                       "value": "December",
//                       "options": [
//                           "January",
//                           "February",
//                           "March",
//                           "April",
//                           "May",
//                           "June",
//                           "July",
//                           "August",
//                           "September",
//                           "October",
//                           "November",
//                           "December"
//                       ],
//                       "description": "Choose any one month of the year.",
//                       "placeholder": "Select"
//                   },
//                   {
//                       "max": 31,
//                       "min": 1,
//                       "name": "backupDate",
//                       "type": "number",
//                       "label": "Backup Date",
//                       "value": "31",
//                       "description": "Enter any one date of the month.",
//                       "placeholder": "Ex: 30"
//                   },
//                   {
//                       "max": 100,
//                       "min": 1,
//                       "name": "alwaysExecutes",
//                       "type": "number",
//                       "label": "Always Executes",
//                       "value": "1",
//                       "description": "Select any number between 1 and 100.",
//                       "placeholder": "Ex: 30"
//                   },
//                   {
//                       "text": "Note: By selecting 'Yearly' as the backup frequency, you need to specify the start time, retention period, executive times and the backup date.",
//                       "type": "note"
//                   }
//               ]
//           }
//       ],
//       "dottedStepperIcon": "./assets/images/paas/dotted-backup-management.svg",
//       "defaultStepperIcon": "./assets/images/paas/default-backup-management.svg",
//       "completedStepperIcon": "./assets/images/paas/complete.svg"
//   },
//   {
//       "label": "Select Add-Ons",
//       "categories": [
//           {
//               "name": "All Tools"
//           },
//           {
//               "name": "Observability"
//           },
//           {
//               "name": "Networking"
//           }
//       ],
//       "description": "Enhance functionality by integrating additional features and services to address specific requirements.",
//       "reviewLabel": "Add-Ons",
//       "formControls": [
//           {
//               "name": "search",
//               "type": "text",
//               "label": "Search",
//               "description": "Search for available add-ons."
//           },
//           {
//               "name": "observability",
//               "type": "observability",
//               "label": "Observability (Optional)",
//               "tools": [
//                   {
//                       "name": "Prometheus + Grafana",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/prometheus.svg",
//                       "imagePath2": "./assets/images/paas/grafana.svg",
//                       "visibility": "APP,MGMT",
//                       "description": "Open-source metrics collection for cloud-native monitoring.",
//                       "mappinginapi": "prometheus",
//                       "moreInfoLink": "https://grafana.com/docs/grafana/latest/dashboards/use-dashboards/"
//                   },
//                   {
//                       "name": "OpenSearch",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/opensearch.svg",
//                       "visibility": "APP,MGMT",
//                       "description": "Scalable search and analytics engine for log and telemetry data.",
//                       "mappinginapi": "opensearch",
//                       "moreInfoLink": "https://docs.opensearch.org/latest/dashboards/quickstart/#using-the-dashboards-application"
//                   },
//                   {
//                       "name": "Kafka",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/opensearch.svg",
//                       "visibility": "MGMT",
//                       "description": "Scalable search and analytics engine for log and telemetry data.",
//                       "mappinginapi": "kafka",
//                       "moreInfoLink": "https://kafka.apache.org/documentation/"
//                   },
//                   {
//                       "name": "ArangoDB",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/opensearch.svg",
//                       "visibility": "MGMT",
//                       "description": "Distributed multi-model database supporting graphs, documents, and key-values.",
//                       "mappinginapi": "arangodb",
//                       "moreInfoLink": "https://www.arangodb.com/docs/stable/"
//                   },
//                   {
//                       "name": "Druid",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/opensearch.svg",
//                       "visibility": "MGMT",
//                       "description": "Real-time analytics database designed for fast slice-and-dice analytics.",
//                       "mappinginapi": "druid",
//                       "moreInfoLink": "https://druid.apache.org/docs/latest/"
//                   },
//                   {
//                       "name": "EFK",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/opensearch.svg",
//                       "visibility": "MGMT",
//                       "description": "Elasticsearch, Fluentd, and Kibana stack for centralized logging.",
//                       "mappinginapi": "efk",
//                       "moreInfoLink": "https://www.elastic.co/what-is/efk-stack"
//                   },
//                   {
//                       "name": "Func Controller",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/opensearch.svg",
//                       "visibility": "MGMT",
//                       "description": "Lightweight controller for managing serverless function workloads.",
//                       "mappinginapi": "func-controller",
//                       "moreInfoLink": "#"
//                   },
//                   {
//                       "name": "Functions PaaS",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/opensearch.svg",
//                       "visibility": "MGMT",
//                       "description": "Platform for deploying and managing serverless functions.",
//                       "mappinginapi": "functions-paas",
//                       "moreInfoLink": "#"
//                   },
//                   {
//                       "name": "GitLab",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/opensearch.svg",
//                       "visibility": "MGMT",
//                       "description": "Complete DevOps platform for source code management and CI/CD.",
//                       "mappinginapi": "gitlab",
//                       "moreInfoLink": "https://docs.gitlab.com/"
//                   },
//                   {
//                       "name": "Harbor",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/opensearch.svg",
//                       "visibility": "MGMT",
//                       "description": "Cloud-native registry for storing, signing, and scanning container images.",
//                       "mappinginapi": "harbor",
//                       "moreInfoLink": "https://goharbor.io/docs/"
//                   },
//                   {
//                       "name": "Knative",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/opensearch.svg",
//                       "visibility": "MGMT",
//                       "description": "Kubernetes-based platform to deploy and manage serverless workloads.",
//                       "mappinginapi": "knative",
//                       "moreInfoLink": "https://knative.dev/docs/"
//                   },
//                   {
//                       "name": "Memcached",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/opensearch.svg",
//                       "visibility": "MGMT",
//                       "description": "High-performance in-memory caching system for dynamic applications.",
//                       "mappinginapi": "memcached",
//                       "moreInfoLink": "https://memcached.org/"
//                   },
//                   {
//                       "name": "Postgres",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/opensearch.svg",
//                       "visibility": "MGMT",
//                       "description": "Powerful open-source relational database system.",
//                       "mappinginapi": "postgres",
//                       "moreInfoLink": "https://www.postgresql.org/docs/"
//                   },
//                   {
//                       "name": "RabbitMQ",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/opensearch.svg",
//                       "visibility": "MGMT",
//                       "description": "Robust messaging broker supporting multiple protocols for distributed systems.",
//                       "mappinginapi": "rabbitmq",
//                       "moreInfoLink": "https://www.rabbitmq.com/docs/"
//                   },
//                   {
//                       "name": "Redis",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/opensearch.svg",
//                       "visibility": "MGMT",
//                       "description": "In-memory data structure store used as a database, cache, and message broker.",
//                       "mappinginapi": "redis",
//                       "moreInfoLink": "https://redis.io/docs/"
//                   },
//                   {
//                       "name": "TCL Ingress Controller",
//                       "type": "toggle",
//                       "value": false,
//                       "imagePath": "./assets/images/paas/opensearch.svg",
//                       "visibility": "MGMT",
//                       "description": "Ingress controller for routing external traffic into Kubernetes clusters.",
//                       "mappinginapi": "tclingress-controller",
//                       "moreInfoLink": "#"
//                   }
//               ],
//               "optional": true,
//               "description": "Track system performance, uptime and errors to maintain seamless operation of your application."
//           }
//       ],
//       "dottedStepperIcon": "./assets/images/paas/dotted-adds-on.svg",
//       "defaultStepperIcon": "./assets/images/paas/default-adds-on.svg",
//       "completedStepperIcon": "./assets/images/paas/complete.svg"
//   },
//   {
//       "label": "Review",
//       "description": "Enhance functionality by integrating additional features and services to address specific requirements.",
//       "formControls": [
//           {
//               "label": "Cluster Configuration",
//               "imagePath": "./assets/images/paas/cluster-config.svg",
//               "stepperLabel": "Cluster Configuration"
//           },
//           {
//               "label": "Networking",
//               "imagePath": "./assets/images/paas/networking.svg",
//               "stepperLabel": "Network Setup"
//           },
//           {
//               "label": "Worker Node Pool",
//               "imagePath": "./assets/images/paas/worker-node-pool.svg",
//               "stepperLabel": "Configuring Worker Node Pool"
//           },
//           {
//               "label": "Persistent Volume Claim",
//               "imagePath": "./assets/images/paas/pvc.svg",
//               "stepperLabel": "Storage Class Configuration",
//               "noDataAvailable": {
//                   "icon": "./assets/images/paas/table-empty-state.svg",
//                   "title": "You haven't added any storage class yet.",
//                   "buttonLabel": "Add Storage Class",
//                   "description": "Storage classes define how storage is provisioned for your workloads. Touse persistent volumes, at least one storage class is required."
//               }
//           },
//           {
//               "label": "Backup",
//               "imagePath": "./assets/images/paas/backup.svg",
//               "stepperLabel": "Backup Management",
//               "arrayInColumn": true,
//               "noDataAvailable": {
//                   "icon": "./assets/images/paas/table-empty-state.svg",
//                   "title": "No Backups Available",
//                   "buttonLabel": "Add Backup Policy",
//                   "description": "No backups have been configured for this system. Please create a backup plan to secure your data."
//               }
//           },
//           {
//               "label": "Add Ons",
//               "imagePath": "./assets/images/paas/add-on.svg",
//               "stepperLabel": "Select Add-Ons"
//           }
//       ],
//       "dottedStepperIcon": "./assets/images/paas/dotted-review.svg",
//       "defaultStepperIcon": "./assets/images/paas/default-review.svg",
//       "completedStepperIcon": "./assets/images/paas/complete.svg"
//   }
// ]
        },
        (error) => {
            console.log("error", error)
        }
    );

  }

  getclusterTemplatecust(){
    this.clusterService.getClusterDetails(3,
    (response) => {
       console.log("response for cluster template", response)
       this.stepDefinitions= response.data
//   this.stepDefinitions = [
//     {
//         "label": "Cluster Configuration",
//         "optional": false,
//         "description": "Configure and manage cluster settings to enhance performance of the application.",
//         "reviewLabel": "Cluster Configuration", 
//         "editInSidenav": true,
//         "formControls": [
//             {
//                 "name": "clusterName",
//                 "type": "text",
//                 "label": "Cluster Name",
//                 "description": "Assign a unique name to your cluster for easy identification",
//                 "validations": {
//                     "message": "Cluster Name should start with a letter and can have only 3 to 18 characters (numbers,alphabets,hyphens and underscores).",
//                     "pattern": "^[a-zA-Z][a-zA-Z0-9-]{2,17}$",
//                     "required": true
//                 },
//                 "customValidation": true
//             },
//             {
//                 "name": "datacenter",
//                 "type": "singleSelect",
//                 "label": "Data Center",
//                 "options": [],
//                 "description": "Select the location where you want your cluster to get deployed .",
//                 "validations": {
//                     "message": "Data Center is required.",
//                     "required": true
//                 }
//             },
//             {
//                 "name": "kubernetesVersion",
//                 "type": "singleSelect",
//                 "label": "Kubernetes version",
//                 "options": [],
//                 "learnmore": "https://kubernetes.io/releases/",
//                 "description": "Pick a Kubernetes version.",
//                 "validations": {
//                     "message": "Kubernetes is required.",
//                     "required": true
//                 }
//             },
//             {
//                 "name": "cnidriver",
//                 "type": "singleSelect",
//                 "label": "Choose a CNI Driver",
//                 "options": [],
//                 "description": "Select the container network interface (CNI) plugin that will manage networking for your pods."
//             },
//             {
//                 "name": "Tags",
//                 "type": "tags",
//                 "label": "Tags",
//                 "options": [],
//                 "learnmore": "https://ipcloud.tatacommunications.com/docs/docs/user-docs/iks/core_concepts#associate-tags",
//                 "description": "Add key-value pairs to define custom labels or settings.",
//                 "validations": null
//             }
//         ],
//         "dottedStepperIcon": "./assets/images/paas/dotted-cluster-configuration.svg",
//         "defaultStepperIcon": "./assets/images/paas/default-cluster-configuration.svg",
//         "completedStepperIcon": "./assets/images/paas/complete.svg"
//     },
//     {
//         "label": "Network Setup",
//         "optional": false,
//         "description": "Configure essential settings for your network.",
//         "reviewLabel": "Networking",
//         "editInSidenav": true,
//         "formControls": [
//             {
//                 "name": "businessUnit",
//                 "type": "singleSelect",
//                 "label": "Business Unit",
//                 "options": [],
//                 "description": "Select the department under which the cluster is to be deployed.",
//                 "validations": {
//                     "message": "Business Unit is required.",
//                     "required": true
//                 },
//                 "addNewOption": true
//             },
//             {
//                 "url": "https://ipcloud.tatacommunications.com/cloud/console/ipc/#/bu/list",
//                 "name": "createbusinessUnit",
//                 "type": "createbutton",
//                 "label": "Create Business Unit"
//             },
//             {
//                 "name": "environment",
//                 "type": "singleSelect",
//                 "label": "Environment",
//                 "options": [],
//                 "learnmore": "https://ipcloud.tatacommunications.com/docs/docs/environments/",
//                 "description": "Choose the deployment stage.",
//                 "validations": {
//                     "message": "Environment is required.",
//                     "required": true
//                 },
//                 "addNewOption": true
//             },
//             {
//                 "url": "https://ipcloud.tatacommunications.com/cloud/console/ipc/#/env/list",
//                 "name": "createenvironment",
//                 "type": "createbutton",
//                 "label": "Create Environment"
//             },
//             {
//                 "name": "zone",
//                 "type": "singleSelect",
//                 "label": "Zone",
//                 "options": [],
//                 "learnmore": "https://ipcloud.tatacommunications.com/docs/docs/zones",
//                 "description": "Specify the logical network(VLAN) for cluster deployment.",
//                 "validations": {
//                     "message": "Zone is required.",
//                     "required": true
//                 },
//                 "addNewOption": true
//             },
//             {
//                 "url": "https://ipcloud.tatacommunications.com/cloud/console/ipc/#/zone/page/list",
//                 "name": "createzone",
//                 "type": "createbutton",
//                 "label": "Create Zone"
//             }
//         ],
//         "dottedStepperIcon": "./assets/images/paas/dotted-Network-Setup.svg",
//         "defaultStepperIcon": "./assets/images/paas/default-Network-Setup.svg",
//         "completedStepperIcon": "./assets/images/paas/complete.svg"
//     },
//     {
//         "type": "table",
//         "label": "Configuring Worker Node Pool",
//         "optional": false,
//         "tableTitle": "List of Worker Node Pool",
//         "description": "Scale your Cluster with Worker Nodes",
//         "reviewLabel": "Worker Node Pool",
//         "formControls": [
//             {
//                 "icon": "./assets/images/paas/os-icon.svg",
//                 "name": "operatingSystem",
//                 "type": "singleSelect",
//                 "label": "Operating System",
//                 "options": [],
//                 "cardPopup": true,
//                 "learnmore": "https://kubernetes.io/docs/concepts/windows/",
//                 "buttonLabel": "Change Operating Environment",
//                 "description": "You are currently using an OS version configured for your worker nodes.",
//                 "imageMapping": {
//                     "SUSE": "./assets/images/os_icons/suse.svg",
//                     "CentOS": "./assets/images/os_icons/centos.svg",
//                     "Ubuntu": "./assets/images/os_icons/ubuntu.svg",
//                     "Windows": "./assets/images/os_icons/windows.svg",
//                     "Oracle Linux": "./assets/images/os_icons/oel.svg",
//                     "Red Hat Enterprise Linux": "./assets/images/os_icons/redhat.svg"
//                 },
//                 "btnDescription": "You are currently using an OS version configured for your worker nodes."
//             },

//             {
//                 "name": "workerNodePoolName",
//                 "type": "text",
//                 "label": "Worker Node Pool Name",
//                 "description": "Assign a name to worker pool node.",
//                 "tableColumn": true,
//                 "validations": {
//                     "message": "Requires only 5 lowercase alphanumeric characters",
//                     "pattern": "^[a-z0-9]{1,5}$",
//                     "required": true
//                 }
//             },
//             {
//                 "name": "node type",
//                 "type": "singleSelect",
//                 "label": "Node Type",
//                 "options": [],
//                 "tableColumn": true,
//                 "validations": {
//                     "message": "Node Type is required.",
//                     "required": true
//                 }
//             },
//             {
//                 "name": "flavour",
//                 "type": "singleSelect",
//                 "label": "Flavour",
//                 "options": [],
//                 "learnmore": "https://ipcloud.tatacommunications.com/docs/docs/user-docs/iks/core_concepts#worker-nodes",
//                 "description": "Select worker pool node compute configuration.",
//                 "tableColumn": true,
//                 "validations": {
//                     "message": "Flavour is required.",
//                     "required": true
//                 }
//             },
//             {
//                 "name": "deploymentScaling",
//                 "type": "number",
//                 "label": "Fixed Replica Count",
//                 "inputType": "number",
//                 "learnmore": "https://ipcloud.tatacommunications.com/docs/docs/user-docs/iks/core_concepts#worker-nodes",
//                 "description": "Keep your app fast and available by adjusting the scale.",
//                 "tableColumn": true,
//                 "validations": {
//                     "max": 8,
//                     "min": 1,
//                     "message": "",
//                     "required": true
//                 }
//             },
//             {
//                 "name": "enable Replica",
//                 "type": "checkbox",
//                 "label": "Enable if you want to auto scale on compute requirements.",
//                 "description": "Keep your app fast and available by adjusting the scale.",
//                 "tableColumn": false
//             },
//             {
//                 "name": "scaledReplica",
//                 "type": "number",
//                 "label": "Scaled Replica Count",
//                 "disabled": false,
//                 "inputType": "number",
//                 "dependency": [
//                     "enable Replica"
//                 ],
//                 "reviewLabel": "Max Replica Count",
//                 "tableColumn": true,
//                 "validations": {
//                     "max": 8,
//                     "min": 1,
//                     "message": ""
//                 }
//             },
//             {
//                 "name": "addWorkerNodePool",
//                 "type": "button",
//                 "label": "Add Worker Node Pool",
//                 "style": "primary",
//                 "disabled": false,
//                 "tableLabel": "List of Worker Node Pools",
//                 "description": "Stay connected with the newest additions to our node list, ensuring seamless operations.",
//                 "discardButton": true
//             }
//         ],
//         "toasterDelete": {
//             "type": "delete",
//             "message": "Worker Node Pool Deleted",
//             "subMessage": "The worker node pool configuration has been deleted"
//         },
//         "toasterSuccess": {
//             "type": "success",
//             "message": "Worker Node Pool Updated",
//             "subMessage": "The worker node pool configuration has been updated"
//         },
//         "tableDescription": "Stay connected with the newest additions to our node list, ensuring seamless operations.",
//         "dottedStepperIcon": "./assets/images/paas/dotted-configure-worker-node-pool.svg",
//         "defaultStepperIcon": "./assets/images/paas/default-configure-worker-node-pool.svg",
//         "completedStepperIcon": "./assets/images/paas/complete.svg",
//         "showFormFieldsOnEdit": true,
//         "hideFormFieldsAfterFirstAdd": true
//     },
//     {
//         "type": "table",
//         "label": "Storage Class Configuration",
//         "optional": true,
//         "tableTitle": "Configured Storage Classes",
//         "description": "This section enables efficient storage configuration for persistent volumes.",
//         "reviewLabel": "Persistent Volume Claim",
//         "formControls": [
//             {
//                 "name": "persistentVolumeName",
//                 "type": "text",
//                 "label": "Persistent Volume Name",
//                 "description": "Provide a name for the storage volume.",
//                 "tableColumn": true,
//                 "validations": {
//                     "message": "Persistent Volume Name is required.",
//                     "required": true
//                 }
//             },
//             {
//                 "name": "classConfiguration",
//                 "type": "singleSelect",
//                 "label": "Class Configuration",
//                 "options": [
//                     {
//                         "id": "IOPS 1",
//                         "itemName": "IOPS 1"
//                     },
//                     {
//                         "id": "IOPS 5",
//                         "itemName": "IOPS 5"
//                     }
//                 ],
//                 "learnmore": "https://ipcloud.tatacommunications.com/docs/docs/user-docs/iks/core_concepts#persistentvolume-pv-and-persistentvolumeclaim-pvc",
//                 "description": "Choose the appropriate IOPS version based on your performance requirements.",
//                 "tableColumn": true,
//                 "validations": {
//                     "message": "Class Configuration is required.",
//                     "required": true
//                 },
//                 "isSelectedValue": [],
//                 "removeSelectedValue": true
//             },
//             {
//                 "name": "capacity",
//                 "type": "number",
//                 "label": "Capacity (GB)",
//                 "learnmore": "https://ipcloud.tatacommunications.com/docs/docs/user-docs/iks/core_concepts#persistentvolume-pv-and-persistentvolumeclaim-pvc",
//                 "description": "Defines the storage capacity in GB, up to a maximum of 7000 GB.",
//                 "tableColumn": true,
//                 "validations": {
//                     "min": 1,
//                     "message": "Capacity is required.",
//                     "required": true
//                 }
//             },
//             {
//                 "name": "addpvc",
//                 "type": "button",
//                 "label": "Add PVC",
//                 "style": "primary",
//                 "disabled": false,
//                 "tableLabel": "Configuring PVCs for Cluster Storage",
//                 "description": "It provides a feature to configuring Persistent Volume Claims (PVCs) to bind to Persistent Volumes (PVs) later.",
//                 "discardButton": true
//             }
//         ],
//         "toasterDelete": {
//             "type": "delete",
//             "message": "Persistent Volume Claim Deleted",
//             "subMessage": "The Storage Class Configuration has been deleted"
//         },
//         "toasterSuccess": {
//             "type": "success",
//             "message": "Storage Class Configuration Updated",
//             "subMessage": "The storage class configuration has been updated"
//         },
//         "tableDescription": "These storage classes define how persistent volumes are provisioned across your cluster.",
//         "dottedStepperIcon": "./assets/images/paas/dotted-storage-class-configuration.svg",
//         "defaultStepperIcon": "./assets/images/paas/default-storage-class-configuration.svg",
//         "completedStepperIcon": "./assets/images/paas/complete.svg",
//         "showFormFieldsOnEdit": true,
//         "hideFormFieldsAfterFirstAdd": true
//     },
//     {
//         "label": "Backup Management",
//         "optional": true,
//         "description": "Take control of your data's security with effective backup management for your cluster.",
//         "reviewLabel": "Backup",
//         "formControls": [
//             {
//                 "info": "",
//                 "name": "backupType",
//                 "type": "backupType",
//                 "label": "Backup Type",
//                 "description": "Choose whether to backup all cluster data or backup specific namespaces only.",
//                 "validations": {
//                     "message": "Backup type is required.",
//                     "required": true
//                 }
//             },
//             {
//                 "name": "byLabels",
//                 "type": "filterbylabel",
//                 "label": "Filter By Labels",
//                 "options": [],
//                 "description": "Label filters help narrow down restore jobs across large environments.",
//                 "validations": {
//                     "message": "required.",
//                     "required": true
//                 }
//             },
//             {
//                 "name": "backupNonNamespacedResources",
//                 "type": "backupNonNamespacedResourcescheckbox",
//                 "label": "Backup non-namespaced resources",
//                 "description": "Backup non-namespaced resources as well"
//             },
//             {
//                 "name": "storagePool",
//                 "type": "storage",
//                 "label": "Storage Pool Configuration",
//                 "description": "Configure storage settings for backup operations.",
//                 "validations": {
//                     "message": "Storage configuration is required.",
//                     "required": true
//                 }
//             },
//             {
//                 "name": "backupFrequency",
//                 "type": "multiSelect",
//                 "label": "Frequency",
//                 "options": [
//                     {
//                         "id": "1",
//                         "itemName": "Daily"
//                     },
//                     {
//                         "id": "2",
//                         "itemName": "Weekly"
//                     },
//                     {
//                         "id": "3",
//                         "itemName": "Monthly"
//                     },
//                     {
//                         "id": "4",
//                         "itemName": "Yearly"
//                     }
//                 ],
//                 "description": "Choose the backup frequency from daily, weekly, monthly, or yearly options."
//             },
//             {
//                 "name": "Daily",
//                 "dependancy": "backupFrequency",
//                 "formControls": [
//                     {
//                         "name": "backupTime",
//                         "type": "time",
//                         "label": "Backup Time",
//                         "value": "22:00",
//                         "description": "Choose the backup frequency from daily, weekly, monthly, or yearly options.",
//                         "placeholder": "00:00:00",
//                         "timezoneLabel": "IST"
//                     },
//                     {
//                         "max": 7,
//                         "min": 1,
//                         "name": "retentionWindow",
//                         "type": "number",
//                         "unit": "Days",
//                         "label": "Retention Window",
//                         "value": "7",
//                         "description": "Set the range between 1 to 7 days",
//                         "placeholder": "Ex: 7"
//                     },
//                     {
//                         "name": "backupType",
//                         "type": "dropdown",
//                         "label": "Backup Type",
//                         "value": "Full",
//                         "options": [
//                             "Full",
//                             "Incremental"
//                         ],
//                         "description": "Select between full or incremental backup.",
//                         "placeholder": "Select"
//                     },
//                     {
//                         "max": 100,
//                         "min": 1,
//                         "name": "alwaysExecutes",
//                         "type": "number",
//                         "label": "Always Executes",
//                         "value": "1",
//                         "description": "Select any number between 1 and 100.",
//                         "placeholder": "Ex: 30"
//                     },
//                     {
//                         "text": "Note: By selecting 'Daily' as the backup frequency, you need to specify the start time, retention period, executive times and the backup type.",
//                         "type": "note"
//                     }
//                 ]
//             },
//             {
//                 "name": "Weekly",
//                 "dependancy": "backupFrequency",
//                 "formControls": [
//                     {
//                         "name": "backupTime",
//                         "type": "time",
//                         "label": "Backup Time",
//                         "value": "22:00",
//                         "description": "Specify the time in hours, minutes, and seconds.",
//                         "placeholder": "00:00:00",
//                         "timezoneLabel": "IST"
//                     },
//                     {
//                         "max": 30,
//                         "min": 7,
//                         "name": "retentionWindow",
//                         "type": "number",
//                         "unit": "Days",
//                         "label": "Retention Window",
//                         "value": "30",
//                         "description": "Set the range between 7 to 30 days.",
//                         "placeholder": "Ex: 18"
//                     },
//                     {
//                         "name": "backupDay",
//                         "type": "dropdown",
//                         "label": "Backup Day",
//                         "value": "Sunday",
//                         "options": [
//                             "Sunday",
//                             "Monday",
//                             "Tuesday",
//                             "Wednesday",
//                             "Thursday",
//                             "Friday",
//                             "Saturday"
//                         ],
//                         "description": "Choose any one day of the week.",
//                         "placeholder": "Select"
//                     },
//                     {
//                         "max": 100,
//                         "min": 1,
//                         "name": "alwaysExecutes",
//                         "type": "number",
//                         "label": "Always Executes",
//                         "value": "1",
//                         "description": "Select any number between 1 and 100.",
//                         "placeholder": "Ex: 30"
//                     },
//                     {
//                         "text": "Note: By selecting 'Weekly' as the backup frequency, you need to specify the start time, retention period, executive times and the backup day.",
//                         "type": "note"
//                     }
//                 ]
//             },
//             {
//                 "name": "Monthly",
//                 "dependancy": "backupFrequency",
//                 "formControls": [
//                     {
//                         "name": "backupTime",
//                         "type": "time",
//                         "label": "Backup Time",
//                         "value": "22:00",
//                         "description": "Specify the time in hours, minutes, and seconds.",
//                         "placeholder": "00:00:00",
//                         "timezoneLabel": "IST"
//                     },
//                     {
//                         "max": 360,
//                         "min": 28,
//                         "name": "retentionWindow",
//                         "type": "number",
//                         "unit": "Days",
//                         "label": "Retention Window",
//                         "value": "90",
//                         "description": "Set the range between 28 to 360 days.",
//                         "placeholder": "Ex: 45"
//                     },
//                     {
//                         "max": 31,
//                         "min": 1,
//                         "name": "backupDate",
//                         "type": "number",
//                         "label": "Backup Date",
//                         "value": 28,
//                         "description": "Enter any one date of the month.",
//                         "placeholder": "Ex: 30"
//                     },
//                     {
//                         "max": 100,
//                         "min": 1,
//                         "name": "alwaysExecutes",
//                         "type": "number",
//                         "label": "Always Executes",
//                         "value": "1",
//                         "description": "Select any number between 1 and 100.",
//                         "placeholder": "Ex: 30"
//                     },
//                     {
//                         "text": "Note: By selecting 'Monthly' as the backup frequency, you need to specify the start time, retention period, executive times and the backup date.",
//                         "type": "note"
//                     }
//                 ]
//             },
//             {
//                 "name": "Yearly",
//                 "dependancy": "backupFrequency",
//                 "formControls": [
//                     {
//                         "name": "backupTime",
//                         "type": "time",
//                         "label": "Backup Time",
//                         "value": "22:00",
//                         "description": "Specify the time in hours, minutes, and seconds.",
//                         "placeholder": "00:00:00",
//                         "timezoneLabel": "IST"
//                     },
//                     {
//                         "max": 1825,
//                         "min": 365,
//                         "name": "retentionWindow",
//                         "type": "number",
//                         "unit": "Days",
//                         "label": "Retention Window",
//                         "value": "730",
//                         "description": "Set the range between 365 to 1825 days",
//                         "placeholder": "Ex: 371"
//                     },
//                     {
//                         "name": "backupMonth",
//                         "type": "dropdown",
//                         "label": "Backup Month",
//                         "value": "December",
//                         "options": [
//                             "January",
//                             "February",
//                             "March",
//                             "April",
//                             "May",
//                             "June",
//                             "July",
//                             "August",
//                             "September",
//                             "October",
//                             "November",
//                             "December"
//                         ],
//                         "description": "Choose any one month of the year.",
//                         "placeholder": "Select"
//                     },
//                     {
//                         "max": 31,
//                         "min": 1,
//                         "name": "backupDate",
//                         "type": "number",
//                         "label": "Backup Date",
//                         "value": "31",
//                         "description": "Enter any one date of the month.",
//                         "placeholder": "Ex: 30"
//                     },
//                     {
//                         "max": 100,
//                         "min": 1,
//                         "name": "alwaysExecutes",
//                         "type": "number",
//                         "label": "Always Executes",
//                         "value": "1",
//                         "description": "Select any number between 1 and 100.",
//                         "placeholder": "Ex: 30"
//                     },
//                     {
//                         "text": "Note: By selecting 'Yearly' as the backup frequency, you need to specify the start time, retention period, executive times and the backup date.",
//                         "type": "note"
//                     }
//                 ]
//             }
//         ],
//         "dottedStepperIcon": "./assets/images/paas/dotted-backup-management.svg",
//         "defaultStepperIcon": "./assets/images/paas/default-backup-management.svg",
//         "completedStepperIcon": "./assets/images/paas/complete.svg"
//     },
//     {
//         "label": "Select Add-Ons",
//         "categories": [
//             {
//                 "name": "All Tools"
//             },
//             {
//                 "name": "Observability"
//             },
//             {
//                 "name": "Networking"
//             }
//         ],
//         "description": "Enhance functionality by integrating additional features and services to address specific requirements.",
//         "reviewLabel": "Add-Ons",
//         "formControls": [
//             {
//                 "name": "search",
//                 "type": "text",
//                 "label": "Search",
//                 "description": "Search for available add-ons."
//             },
//             {
//                 "name": "observability",
//                 "type": "observability",
//                 "label": "Observability (Optional)",
//                 "tools": [
//                     {
//                         "name": "Prometheus + Grafana",
//                         "type": "toggle",
//                         "value": false,
//                         "imagePath": "./assets/images/paas/prometheus.svg",
//                         "imagePath2": "./assets/images/paas/grafana.svg",
//                         "visibility": "APP,MGMT",
//                         "description": "Open-source metrics collection for cloud-native monitoring.",
//                         "mappinginapi": "prometheus",
//                         "moreInfoLink": "https://grafana.com/docs/grafana/latest/dashboards/use-dashboards/"
//                     },
//                     {
//                         "name": "OpenSearch",
//                         "type": "toggle",
//                         "value": false,
//                         "imagePath": "./assets/images/paas/opensearch.svg",
//                         "visibility": "APP,MGMT",
//                         "description": "Scalable search and analytics engine for log and telemetry data.",
//                         "mappinginapi": "opensearch",
//                         "moreInfoLink": "https://docs.opensearch.org/latest/dashboards/quickstart/#using-the-dashboards-application"
//                     },
//                     {
//                         "name": "Kafka",
//                         "type": "toggle",
//                         "value": false,
//                         "imagePath": "./assets/images/paas/opensearch.svg",
//                         "visibility": "MGMT",
//                         "description": "Scalable search and analytics engine for log and telemetry data.",
//                         "mappinginapi": "kafka",
//                         "moreInfoLink": "https://kafka.apache.org/documentation/"
//                     }
//                 ],
//                 "optional": true,
//                 "description": "Track system performance, uptime and errors to maintain seamless operation of your application."
//             }
//         ],
//         "dottedStepperIcon": "./assets/images/paas/dotted-adds-on.svg",
//         "defaultStepperIcon": "./assets/images/paas/default-adds-on.svg",
//         "completedStepperIcon": "./assets/images/paas/complete.svg"
//     },
//     {
//         "label": "Review",
//         "description": "Enhance functionality by integrating additional features and services to address specific requirements.",
//         "formControls": [
//             {
//                 "label": "Cluster Configuration",
//                 "imagePath": "./assets/images/paas/cluster-config.svg",
//                 "stepperLabel": "Cluster Configuration",               
//             },
//             {
//                 "label": "Networking",
//                 "imagePath": "./assets/images/paas/networking.svg",
//                 "stepperLabel": "Network Setup",
//             },
//             {
//                 "label": "Worker Node Pool",
//                 "imagePath": "./assets/images/paas/worker-node-pool.svg",
//                 "stepperLabel": "Configuring Worker Node Pool"
//             },
//             {
//                 "label": "Persistent Volume Claim",
//                 "imagePath": "./assets/images/paas/pvc.svg",
//                 "stepperLabel": "Storage Class Configuration"
//             },
//             {
//                 "label": "Backup",
//                 "imagePath": "./assets/images/paas/backup.svg",
//                 "stepperLabel": "Backup Management",
//                 "arrayInColumn": true
//             },
//             {
//                 "label": "Add Ons",
//                 "imagePath": "./assets/images/paas/add-on.svg",
//                 "stepperLabel": "Select Add-Ons"
//             }
//         ],
//         "dottedStepperIcon": "./assets/images/paas/dotted-review.svg",
//         "defaultStepperIcon": "./assets/images/paas/default-review.svg",
//         "completedStepperIcon": "./assets/images/paas/complete.svg"
//     }
// ]
    },
    (error) => {
        console.log("error", error)
    }
);

}


  addons: any = {
    data: []
  };
  kubernetesVersion: string = '';
  onKubernetesVersionChanged(version: string) {
    // Prevent duplicate calls for the same version
    if (this.lastKubernetesVersion === version) {
      // console.log(`Duplicate kubernetes version change detected: ${version}, ignoring`);
      return;
    }
    
    this.lastKubernetesVersion = version;
    this.kubernetesVersion = version;
    // console.log("version of kubernetes from parent", this.kubernetesVersion)
    
    // For customer login, hardcode clusterType to "APP"
    if (this.flags.isCustomer && !this.clusterType) {
      this.clusterType = "APP";
      // console.log("Customer login detected, setting clusterType to APP");
    }
    
    // Call getnetworklist when kubernetes version changes (if we have all required parameters)
    if (this.selectedDataCenterEndpointId && this.kubernetesVersion && this.clusterType) {
      // console.log('All required parameters available, calling getnetworklist');
      this.getnetworklist(this.selectedDataCenterEndpointId);
    } else {
      // console.log('Missing required parameters for getnetworklist:');
      // console.log('- selectedDataCenterEndpointId:', this.selectedDataCenterEndpointId);
      // console.log('- kubernetesVersion:', this.kubernetesVersion);
      // console.log('- clusterType:', this.clusterType);
    }
    
    // Call getaddons when kubernetes version changes (if we have all required parameters)
    if (this.zoneId && this.kubernetesVersion && this.clusterType) {
      // console.log('All required parameters available, calling getaddons');
      this.getaddons();
    } else {
      // console.log('Missing required parameters for getaddons:');
      // console.log('- zoneId:', this.zoneId);
      // console.log('- kubernetesVersion:', this.kubernetesVersion);
      // console.log('- clusterType:', this.clusterType);
    }
    
    // Refresh OS filtering when kubernetes version changes (if we have zoneId)
    if (this.zoneId) {
      // console.log('Refreshing OS filtering with new kubernetes version:', this.kubernetesVersion);
      this.loadOSAndFlavours();
    } else {
      // console.log('Missing zoneId for OS filtering refresh');
    }
  }
  onEngagementChange(eng: any) {
    // Extract ID from the object if it's a full object
    this.selectedEngId = eng.value?.id || eng.value;
    this.callEngagementDependentData(this.selectedEngId);
  }
  // onZoneChange(zone: any) {
  //   console.log("enetered on zone", zone.value.id)
  //   this.zoneId = zone.value.id
  //   if (this.zoneId) {
  //     if (this.zoneId) {
  //       this.loadOSAndFlavours();
  //     }
  //   }
  // }
  onZoneChange(zone: any) {
    console.log("ZoneLsi", zone);

    // Extract ID from the object if it's a full object
    this.zoneId = zone.value?.id || zone.value;
    
    // For customer login, hardcode clusterType to "APP"
    if (this.flags.isCustomer && !this.clusterType) {
      this.clusterType = "APP";
      // console.log("Customer login detected, setting clusterType to APP");
    }
    
    if (this.zoneId) {
      // Load OS and flavours using the existing method
      this.loadOSAndFlavours();
      
      // Call getaddons when zone changes (if we have all required parameters)
      if (this.zoneId && this.kubernetesVersion && this.clusterType) {
        console.log('All required parameters available, calling getaddons');
        this.getaddons();
      } else {
        // console.log('Missing required parameters for getaddons:');
        // console.log('- zoneId:', this.zoneId);
        // console.log('- kubernetesVersion:', this.kubernetesVersion);
        // console.log('- clusterType:', this.clusterType);
      }
    }
    
  }


  loadOSAndFlavours() {
    forkJoin([
      this.getostype$(),
      this.getflavours$()
    ]).subscribe({
      next: ([osTypes, flavours]) => {
        this.ostype = osTypes;
        this.flavours = flavours;

        if (this.selectedOSVersion) {
          this.onOSTypeSelect(this.selectedOSVersion);
        }
      },
      error: (err) => {
        console.error('Error loading OS and flavours:', err);
      }
    });
  }

  onBusinessUnitChange(departmentId: string) {
    const selectedBU = this.businessUnitOptions.find(bu => bu.id === departmentId);
    this.selectedBusinessUnitName = selectedBU?.itemName || '';

    const filteredEnvs = this.environmentOptions
      .filter(env => env.departmentId === departmentId)
      .map(env => ({
        id: env.id,
        itemName: env.itemName
      }));

    const networkSetupStep = this.stepDefinitions.find(step => step.label === 'Network Setup');
    
    if (networkSetupStep) {
      // Update environment options
      const environmentControl = networkSetupStep.formControls.find(ctrl => ctrl.name === 'environment');
      
      if (environmentControl) {
        environmentControl.options = filteredEnvs;
      }

      // Clear zones options when business unit changes
      const zoneControl = networkSetupStep.formControls.find(ctrl => ctrl.name === 'zone');
      
      if (zoneControl) {
        zoneControl.options = [];
      }

      // Clear environment value when business unit changes
      if (environmentControl) {
        environmentControl.value = null;
        this.selectedEnvironmentName = '';
      }
    }
  }

  onEnvironmentChange(environmentName: string) {
    this.selectedEnvironmentName = environmentName;

    const filteredZones = this.zoneslist
      .filter(zone =>
        zone.departmentName === this.selectedBusinessUnitName &&
        zone.environmentName === this.selectedEnvironmentName
      )
      .map(zone => ({
        id: zone.zoneId,
        itemName: zone.zoneName
      }));

    const networkSetupStep = this.stepDefinitions.find(step => step.label === 'Network Setup');
    
    if (networkSetupStep) {
      const zoneControl = networkSetupStep.formControls.find(ctrl => ctrl.name === 'zone');
      
      if (zoneControl) {
        zoneControl.options = filteredZones;
        // Clear zone value when environment changes
        zoneControl.value = null;
      } else {
        console.warn('Zone control not found in stepDefinitions');
      }
    } else {
      console.warn('Network Setup step not found in stepDefinitions');
    }
  }

  onClusterTypeChange(clusterType: string) {
    this.clusterType = clusterType;
    console.log('Cluster type changed to:', clusterType);
    
    // For customer login, hardcode clusterType to "APP"
    if (this.flags.isCustomer && !this.clusterType) {
      this.clusterType = "APP";
      console.log("Customer login detected, setting clusterType to APP");
    }
    
    // Call getnetworklist when cluster type changes (if we have all required parameters)
    if (this.selectedDataCenterEndpointId && this.kubernetesVersion && this.clusterType) {
      console.log('All required parameters available, calling getnetworklist');
      this.getnetworklist(this.selectedDataCenterEndpointId);
    } else {
      // console.log('Missing required parameters for getnetworklist:');
      // console.log('- selectedDataCenterEndpointId:', this.selectedDataCenterEndpointId);
      // console.log('- kubernetesVersion:', this.kubernetesVersion);
      // console.log('- clusterType:', this.clusterType);
    }
    
    // Call getaddons when cluster type changes (if we have all required parameters)
    if (this.zoneId && this.kubernetesVersion && this.clusterType) {
      console.log('All required parameters available, calling getaddons');
      this.getaddons();
    } else {
      // console.log('Missing required parameters for getaddons:');
      // console.log('- zoneId:', this.zoneId);
      // console.log('- kubernetesVersion:', this.kubernetesVersion);
      // console.log('- clusterType:', this.clusterType);
    }
  }

  onDataCenterChange(selectedDataCenter: any) {
    if (selectedDataCenter && selectedDataCenter.endpointId) {
      // console.log('Data Center selected:', selectedDataCenter);
      // console.log('EndpointId:', selectedDataCenter.endpointId);
      // console.log('EndpointMap:', selectedDataCenter.endpointmap);
      
      // Store the selected data center endpointId
      this.selectedDataCenterEndpointId = selectedDataCenter.endpointId;
      
      // Store the selected endpointmap for filtering environments
      this.selectedDataCenterEndpointMap = selectedDataCenter.endpointmap;
      
      // Check which categories the selected data center exists in
      this.selectedDataCenterCategoryStatus = this.getDataCenterCategoryStatus(selectedDataCenter.endpointId);
      this.isSelectedDataCenterVCPEnabled = this.selectedDataCenterCategoryStatus.inVksEnabled;
      console.log(`Data center ${selectedDataCenter.endpointName} (ID: ${selectedDataCenter.endpointId}) category status:`, this.selectedDataCenterCategoryStatus);
      
      // Update control plane type options based on VCP status
      this.updateControlPlaneTypeOptions();
      
      // Clear previous selections when data center changes
      this.selectedBusinessUnitName = '';
      this.selectedEnvironmentName = '';
      
      // Clear network setup dropdowns when data center changes
      this.clearNetworkSetupDropdowns();
      
      // For customer login, hardcode clusterType to "APP"
      if (this.flags.isCustomer && !this.clusterType) {
        this.clusterType = "APP";
        console.log("Customer login detected, setting clusterType to APP");
      }
      
      // Call getiksimageversions with the selected endpointId
      this.getiksimageversions(selectedDataCenter.endpointId);
      
      // Call getenvironment with the selected endpointmap to filter environments
      if (this.selectedDataCenterEndpointMap) {
        this.getenvironment(this.selectedDataCenterEndpointMap);
      }
      
      // Call getnetworklist when data center changes (if we have all required parameters)
      if (this.selectedDataCenterEndpointId && this.kubernetesVersion && this.clusterType) {
        console.log('All required parameters available, calling getnetworklist');
        this.getnetworklist(this.selectedDataCenterEndpointId);
      } else {
        // console.log('Missing required parameters for getnetworklist:');
        // console.log('- selectedDataCenterEndpointId:', this.selectedDataCenterEndpointId);
        // console.log('- kubernetesVersion:', this.kubernetesVersion);
        // console.log('- clusterType:', this.clusterType);
      }
    }
  }

  // Method to update control plane type options based on category status
  private updateControlPlaneTypeOptions() {
    this.stepDefinitions.forEach((step) => {
      step.formControls.forEach((control) => {
        if (control.name === 'controlPlaneType') {
          // Determine which options to grey out based on category presence
          const { inVksEnabled, inAllImages } = this.selectedDataCenterCategoryStatus;
          
          // Logic:
          // - If in BOTH vks-enabledImages AND all-images → No grey out (both available)
          // - If ONLY in all-images (not in vks-enabledImages) → Grey out "virtual"
          // - If ONLY in vks-enabledImages (not in all-images) → Grey out "dedicated"
          
          control.options = control.options.map((option: any) => {
            let shouldGreyOut = false;
            
            if (option.id === 'virtual') {
              // Grey out virtual if data center is ONLY in all-images (not in vks-enabledImages)
              shouldGreyOut = inAllImages && !inVksEnabled;
            }
            
            return {
              ...option,
              disabled: shouldGreyOut,
              greyedOut: shouldGreyOut
            };
          });
          
          // Clear selection if the currently selected option is now greyed out
          const currentValue = control.value;
          const selectedOption = control.options.find((opt: any) => opt.id === currentValue);
          if (selectedOption && (selectedOption.disabled || selectedOption.greyedOut)) {
            // Find the first non-greyed out option
            const availableOption = control.options.find((opt: any) => !opt.disabled && !opt.greyedOut);
            if (availableOption) {
              control.value = availableOption.id;
              // console.log(`Cleared ${currentValue} selection, switched to ${availableOption.id} due to category restrictions`);
            }
          }
          
          // console.log('Updated control plane type options based on category status:', {
          //   categoryStatus: this.selectedDataCenterCategoryStatus,
          //   options: control.options
          // });
        }
      });
    });
  }

  filterObservabilityTools(addons: any) {
    // Extract the addon names from the response
    const addonNames = addons?.data || [];
    // console.log("entered filterobservabilitytools function")
    // console.log("Available addon names:", addonNames);

    // Find the observability step in stepDefinitions
    const observabilityStep = this.stepDefinitions.find(step => 
      step.label === 'Select Add-Ons' || step.formControls?.some(control => control.type === 'observability')
    );

    if (observabilityStep) {
      const observabilityControl = observabilityStep.formControls.find(control => control.type === 'observability');
      
      if (observabilityControl && observabilityControl.tools) {
        // Store original tools if not already stored
        if (!observabilityControl.originalTools) {
          observabilityControl.originalTools = JSON.parse(JSON.stringify(observabilityControl.tools));
          // console.log("Stored original observability tools:", observabilityControl.originalTools);
        }
        
        // console.log("Original observability tools:", observabilityControl.originalTools);
        
        // Define conditions
        const conditionA = (this.flags.isEngineer && this.clusterType === 'APP') || this.flags.isCustomer;
        const conditionB = this.flags.isEngineer && this.clusterType === 'MGMT';
        
        // console.log("Condition A (Engineer+APP or Customer):", conditionA);
        // console.log("Condition B (Engineer+MGMT):", conditionB);
        
        // Filter tools based on visibility and available addons using mappinginapi values
        const filteredTools = observabilityControl.originalTools.filter(tool => {
          // First check visibility based on conditions
          let shouldShowByVisibility = false;
          
          if (tool.visibility) {
            const visibilityArray = tool.visibility.split(',').map(v => v.trim());
            
            if (conditionA && (visibilityArray.includes('APP') || visibilityArray.includes('APP,MGMT'))) {
              shouldShowByVisibility = true;
            } else if (conditionB && (visibilityArray.includes('MGMT') || visibilityArray.includes('APP,MGMT'))) {
              shouldShowByVisibility = true;
            }
          } else {
            // If no visibility specified, show by default
            shouldShowByVisibility = true;
          }
          
          if (!shouldShowByVisibility) {
            // console.log(`${tool.name} filtered out by visibility (${tool.visibility})`);
            return false;
          }
          
          // Then check mappinginapi if tool should be shown by visibility
          if (tool.mappinginapi) {
            const mappingKeyword = tool.mappinginapi.toLowerCase();
            const shouldShowByMapping = addonNames.some((addon: string) => 
              addon.toLowerCase().includes(mappingKeyword)
            );
            // console.log(`${tool.name} (visibility: ${tool.visibility}, mapping: ${mappingKeyword}) should show: ${shouldShowByMapping}`);
            return shouldShowByMapping;
          }
          
          // For tools without mappinginapi, show by default
         // console.log(`${tool.name} (visibility: ${tool.visibility}, no mapping) should show: true (default)`);
          return true;
        });

        // Update the tools array with filtered results
        observabilityControl.tools = filteredTools;
        // console.log("Filtered observability tools:", observabilityControl.tools);
      }
    }
  }



  onCreateLinkClicked(): void {
   // console.log('[onCreateLinkClicked] Setting refresh flag');
    sessionStorage.setItem('shouldRefreshDropdowns', 'true');
  }


  onDropdownClick(): void {
    const shouldRefresh = sessionStorage.getItem('shouldRefreshDropdowns') === 'true';
   // console.log('[onDropdownClick] Should refresh dropdowns:', shouldRefresh);

    if (shouldRefresh) {
    //  console.log('[onDropdownClick] Refreshing dropdowns...');
      // Call getenvironment only if datacenter is selected
      if (this.selectedDataCenterEndpointMap) {
        this.getenvironment(this.selectedDataCenterEndpointMap);
      }
      // Don't call getenvironment if no datacenter is selected
      this.getzones();
      sessionStorage.removeItem('shouldRefreshDropdowns');

    }
  }


  onStepChange(event: StepperSelectionEvent): void {
    this.currentStepper = event.selectedIndex;
  }

  goToNextStep(stepper: MatStepper): void {
    stepper.next();
    this.currentStepper = stepper.selectedIndex;
  }

  goToPreviousStep(stepper: MatStepper): void {
    if (this.currentStepper > 0) {
      this.currentStepper = this.currentStepper - 1
    }
  }

  handleChildData(data: any) {

    if (this.flags.isEngineer) {
    console.log('Received from child:', data);
    console.log(this.addons,"addons in payload")
    const name = data.clusterData["Cluster Configuration"].clusterName;
    const vmpurpose = data.clusterData["Cluster Configuration"].clusterType;
    // Store the cluster type for use in getaddons method
    this.clusterType = vmpurpose;
    const imageid = data.clusterData["Configuring Worker Node Pool"].formControls[0].options[0].osId;
    const zoneid = this.zoneId;
    //data.clusterData["Network Setup"].businessUnit[0].id;

    const tags = data.clusterData["Cluster Configuration"]?.Tags?.map(item => ({
      id: item.id,
      description: item.value,
      name: item.key
    })) || [];

    const dedicatedDeploy = data.clusterData["Cluster Configuration"]?.controlPlaneType;

    let clusterMode: string;
    let dedicatedDeployment: boolean = false;

    if (vmpurpose === "APP") {
      if (dedicatedDeploy === "dedicated") {
        clusterMode = data.clusterData["Cluster Configuration"].dedicatedControlPlaneType?.id || "Single Master";
        dedicatedDeployment = true;
      } else {
        clusterMode = "High availability";
      }
    } else if (vmpurpose === "MGMT") {
      clusterMode = "High availability";
      dedicatedDeployment = dedicatedDeploy === "dedicated";
    }

    let masterNode;

    if (dedicatedDeploy === "dedicated") {
      const masterFormControl =
      data?.clusterData["Configuring Worker Node Pool"]?.masterFlavor
    
      masterNode = {
        vmHostName: "",
        vmFlavor: masterFormControl?.flavorname || "B4",
        skuCode: masterFormControl?.flavorskucode || "B4.UBN",
        nodeType: "Master",
        replicaCount: vmpurpose === "MGMT" ? 3 : 1,
        maxReplicaCount: masterFormControl?.scaledReplica ?? null,
        additionalDisk: {},
        // flavorDisk: masterFormControl?.options?.[0]?.flavordisk || 50,        
        labelsNTaints: "no"
      };
    } else {
      masterNode = {
        vmHostName: "",
        vmFlavor: "D8",
        skuCode: "D8.UBN",
        nodeType: "Master",
        replicaCount: 3,
        maxReplicaCount: null,
        additionalDisk: {},
        // flavorDisk: 100,
        labelsNTaints: "no"
      };
    }
    

    const rawWorkers = data.clusterData["Configuring Worker Node Pool"].originalOptionsData || [];

    const workerNodes = rawWorkers.map(item => ({
      vmHostName: item.workerNodePoolName,
      vmFlavor: item.flavour?.flavorname || "",
      skuCode: item.flavour?.flavorskucode || "",
      nodeType: "Worker",
      replicaCount: item.deploymentScaling,
      maxReplicaCount: item.scaledReplica ?? null,
      additionalDisk: {},
      // flavorDisk: item.flavour?.flavordisk || 0,
      labelsNTaints: "no" 
    }));
    //Fix the payloads for the dropdown values................
    const vmSpecificInput = [masterNode, ...workerNodes];
    const k8sversion = data.clusterData["Cluster Configuration"]?.kubernetesVersion?.itemName
    const valueosmodel = data.clusterData["Configuring Worker Node Pool"].formControls[0].options[0].osModel
    const valueosmake = data.clusterData["Configuring Worker Node Pool"].formControls[0].options[0].osMake
    const valueosversion = data.clusterData["Configuring Worker Node Pool"].formControls[0].options[0].osVersion
    const Hypervisor = data.clusterData["Configuring Worker Node Pool"].formControls[0].options[0].hypervisor
    const backupData = data.clusterData["Backup Management"] || {};

    let iksBackupDetails: any = null;

    if (backupData?.storagePool?.storagePoolName) {
      iksBackupDetails = {
        storagePoolName: backupData.storagePool.storagePoolName || "", 
        storagePoolId: backupData.storagePool.storagePoolId || "",
        backupNonNamespace: backupData.backupNonNamespacedResources || false,
        capacity: backupData?.fetcapacity || 1,
        modelType: "monthlyReserve",
        daily: backupData.Daily ? {
          startTime: backupData.Daily.backupTime || "22:00",
          retentionWindow: backupData.Daily.retentionWindow || "7",
          runsEvery: backupData.Daily.alwaysExecutes || "1",
          backupType: backupData.Daily.backupType?.toUpperCase() || "FULL"
        } : {
          startTime: "22:00",
          retentionWindow: "7",
          runsEvery: "1",
          backupType: "FULL"
        },
    
        weekly: backupData.Weekly ? {
          startTime: backupData.Weekly.backupTime || "22:00",
          retentionWindow: backupData.Weekly.retentionWindow || "30",
          runsEvery: backupData.Weekly.alwaysExecutes || "1",
          backupDay: backupData.Weekly.backupDay?.toUpperCase() || "SUNDAY"
        } : {
          startTime: "22:00",
          retentionWindow: "30",
          runsEvery: "1",
          backupDay: "SUNDAY"
        },
    
        monthly: backupData.Monthly ? {
          startTime: backupData.Monthly.backupTime || "22:00",
          retentionWindow: backupData.Monthly.retentionWindow || "90",
          runsEvery: backupData.Monthly.alwaysExecutes || "1",
          backupDate: backupData.Monthly.backupDate || "28"
        } : {
          startTime: "22:00",
          retentionWindow: "90",
          runsEvery: "1",
          backupDate: "28"
        },
    
        yearly: backupData.Yearly ? {
          startTime: backupData.Yearly.backupTime || "22:00",
          retentionWindow: backupData.Yearly.retentionWindow || "730",
          backupDate: backupData.Yearly.backupDate || "31",
          backupMonth: backupData.Yearly.backupMonth?.toUpperCase() || "DECEMBER"
        } : {
          startTime: "22:00",
          retentionWindow: "730",
          backupDate: "31",
          backupMonth: "DECEMBER"
        }
      };

      // Add conditional labels or namespaces based on backupType
      if (backupData.backupType === 'fullClusterBackup') {
        iksBackupDetails.namespaces = ["ALL"];
        iksBackupDetails.backupNonNamespace = true
      } else {
        iksBackupDetails.labels = backupData.byLabels;
      }
    }
    const rawPVCs = data.tableContent["Storage Class Configuration"]?.data.rawData || [];

    const addOnSelections = data.clusterData["Select Add-Ons"] || [];
    const cniDriver = data.clusterData["Cluster Configuration"]?.cnidriver?.id;
const managedServices: { name: string }[] = [];

if (this.addons?.data?.length > 0) {
  // Find the observability step in stepDefinitions to get tool mappings
  const observabilityStep = this.stepDefinitions.find(step => 
    step.label === 'Select Add-Ons' || step.formControls?.some(control => control.type === 'observability')
  );

  if (observabilityStep) {
    const observabilityControl = observabilityStep.formControls.find(control => control.type === 'observability');
    
    if (observabilityControl && observabilityControl.originalTools) {
      // Iterate through each selected addon in addOnSelections
      addOnSelections.forEach(selectedAddon => {
        // Get the tool name from the selected addon object
        const toolName = Object.keys(selectedAddon)[0];
        const isSelected = selectedAddon[toolName];
        
        if (isSelected) {
          // Find the corresponding tool in originalTools
          const tool = observabilityControl.originalTools.find(t => t.name === toolName);
          
          if (tool && tool.mappinginapi) {
            // Find the addon entry that matches the mappinginapi value
            const addonEntry = this.addons.data.find(entry => 
              entry.toLowerCase().includes(tool.mappinginapi.toLowerCase())
            );
            if (addonEntry) {
              managedServices.push({ name: addonEntry });
              console.log(`Added ${toolName} (${tool.mappinginapi}) as ${addonEntry} to managedServices`);
            }
          }
        }
      });
    }
  }
}
   //payload for eng
    let payload = {
      "name": "",
      "hypervisor": Hypervisor,
      "purpose": "ipc",
      "vmPurpose": vmpurpose,
      // flavorId: dedicatedDeploy === "dedicated" ? 3234 : 3261,
      "imageId": imageid,
      "zoneId": zoneid,
      "alertSuppression": true,
      "iops": 1,
      "isKdumpOrPageEnabled": "No",
      ...(tags.length > 0 && { tags }),

      "applicationType": "Container",
      "application": "Containers",
      vmSpecificInput,
      clusterMode,
      dedicatedDeployment,
      "clusterName": name,
      "k8sVersion": k8sversion,
      "circuitId": this.getCopfId(),
      "vApp": "",
      "imageDetails": {
        "valueOSModel": valueosmodel,
        "valueOSMake": valueosmake,
        "valueOSVersion": valueosversion,
        "valueOSServicePack": null
      },
      // "flavorDisk": 50,
      ...(iksBackupDetails && { iksBackupDetails }),
      // ✅ Inline conditional property
      ...(rawPVCs.length > 0 && {
        pvcsEnable: rawPVCs.map(item => ({
          size: item.capacity,
          iops: parseInt(item.classConfiguration?.replace(/\D/g, ''), 10) || null
        }))
      }),
      ...(managedServices.length > 0 && { managedServices }),
      ...(cniDriver && {
        networkingDriver: [
          {
            name: cniDriver
          }
        ]
      })
    }
    console.log("final payload", payload)
    console.log("Final Payload in json", JSON.stringify(payload, null, 2));
    this.getLoadingModal("Initiating Instance launch ...");

      this.getLaunchVmData(payload);
    }
  else{
    console.log('Received from child:', data);
    const name = data.clusterData["Cluster Configuration"].clusterName;
    const vmpurpose = data.clusterData["Cluster Configuration"].clusterType;
    const imageid = data.clusterData["Configuring Worker Node Pool"].formControls[0].options[0].osId;
    const zoneid = this.zoneId;
    //data.clusterData["Network Setup"].businessUnit[0].id;

    const tags = data.clusterData["Cluster Configuration"]?.Tags?.map(item => ({
      id: item.id,
      description: item.value,
      name: item.key
    })) || [];

    const dedicatedDeploy = data.clusterData["Cluster Configuration"]?.controlPlaneType;

    let clusterMode: string;
    let dedicatedDeployment: boolean = false;

    if (vmpurpose === "APP") {
      if (dedicatedDeploy === "dedicated") {
        clusterMode = data.clusterData["Cluster Configuration"].dedicatedControlPlaneType?.id || null;
        dedicatedDeployment = true;
      } else {
        clusterMode = "High availability";
      }
    } else if (vmpurpose === "MGMT") {
      clusterMode = "High availability";
      dedicatedDeployment = dedicatedDeploy === "dedicated";
    }

    let masterNode;


      masterNode = {
        vmHostName: "",
        vmFlavor: "D8",
        skuCode: "D8.UBN",
        nodeType: "Master",
        replicaCount: 3,
        maxReplicaCount: null,
        additionalDisk: {},
        // flavorDisk: 100,
        labelsNTaints: "no"
      };
    
    

    const rawWorkers = data.clusterData["Configuring Worker Node Pool"].originalOptionsData || [];

    const workerNodes = rawWorkers.map(item => ({
      vmHostName: item.workerNodePoolName,
      vmFlavor: item.flavour?.flavorname || "B4",
      skuCode: item.flavour?.flavorskucode || "B4.UBN",
      nodeType: "Worker",
      replicaCount: item.deploymentScaling,
      maxReplicaCount: item.scaledReplica ?? null,
      additionalDisk: {},
      // flavorDisk: item.flavour?.flavordisk || 0,
      labelsNTaints: "no"
    }));
    //Fix the payloads for the dropdown values................
    const vmSpecificInput = [masterNode, ...workerNodes];
    const k8sversion = data.clusterData["Cluster Configuration"]?.kubernetesVersion?.id
    const valueosmodel = data.clusterData["Configuring Worker Node Pool"].formControls[0].options[0].osModel
    const valueosmake = data.clusterData["Configuring Worker Node Pool"].formControls[0].options[0].osMake
    const valueosversion = data.clusterData["Configuring Worker Node Pool"].formControls[0].options[0].osVersion
    const Hypervisor = data.clusterData["Configuring Worker Node Pool"].formControls[0].options[0].hypervisor
    const backupData = data.clusterData["Backup Management"] || {};
    const cniDriver = data.clusterData["Cluster Configuration"]?.cnidriver?.id;

    let iksBackupDetails: any = null;

    if (backupData?.storagePool?.storagePoolName) {
      // Initialize the base payload
      iksBackupDetails = {
        storagePoolName: backupData.storagePool.storagePoolName || "", 
        storagePoolId: backupData.storagePool.storagePoolId || "",
        backupNonNamespace: backupData.backupNonNamespacedResources || false,
        capacity: backupData?.fetcapacity || 1,
        modelType: "monthlyReserve",
        daily: backupData.Daily ? {
          startTime: backupData.Daily.backupTime || "22:00",
          retentionWindow: backupData.Daily.retentionWindow || "7",
          runsEvery: backupData.Daily.alwaysExecutes || "1",
          backupType: backupData.Daily.backupType?.toUpperCase() || "FULL"
        } : {
          startTime: "22:00",
          retentionWindow: "7",
          runsEvery: "1",
          backupType: "FULL"
        },
    
        weekly: backupData.Weekly ? {
          startTime: backupData.Weekly.backupTime || "22:00",
          retentionWindow: backupData.Weekly.retentionWindow || "30",
          runsEvery: backupData.Weekly.alwaysExecutes || "1",
          backupDay: backupData.Weekly.backupDay?.toUpperCase() || "SUNDAY"
        } : {
          startTime: "22:00",
          retentionWindow: "30",
          runsEvery: "1",
          backupDay: "SUNDAY"
        },
    
        monthly: backupData.Monthly ? {
          startTime: backupData.Monthly.backupTime || "22:00",
          retentionWindow: backupData.Monthly.retentionWindow || "90",
          runsEvery: backupData.Monthly.alwaysExecutes || "1",
          backupDate: backupData.Monthly.backupDate || "28"
        } : {
          startTime: "22:00",
          retentionWindow: "90",
          runsEvery: "1",
          backupDate: "28"
        },
    
        yearly: backupData.Yearly ? {
          startTime: backupData.Yearly.backupTime || "22:00",
          retentionWindow: backupData.Yearly.retentionWindow || "730",
          backupDate: backupData.Yearly.backupDate || "31",
          backupMonth: backupData.Yearly.backupMonth?.toUpperCase() || "DECEMBER"
        } : {
          startTime: "22:00",
          retentionWindow: "730",
          backupDate: "31",
          backupMonth: "DECEMBER"
        }
      };

      // Add conditional labels or namespaces based on backupType
      if (backupData.backupType === 'fullClusterBackup') {
        iksBackupDetails.namespaces = ["ALL"];
      } else {
        iksBackupDetails.labels = backupData.byLabels;
      }
    }
    const rawPVCs = data.tableContent["Storage Class Configuration"]?.data.rawData || [];

    const addOnSelections = data.clusterData["Select Add-Ons"] || [];
    const managedServices: { name: string }[] = [];
    
    if (this.addons?.data?.length > 0) {
      // Find the observability step in stepDefinitions to get tool mappings
      const observabilityStep = this.stepDefinitions.find(step => 
        step.label === 'Select Add-Ons' || step.formControls?.some(control => control.type === 'observability')
      );
    
      if (observabilityStep) {
        const observabilityControl = observabilityStep.formControls.find(control => control.type === 'observability');
        
        if (observabilityControl && observabilityControl.originalTools) {
          // Iterate through each selected addon in addOnSelections
          addOnSelections.forEach(selectedAddon => {
            // Get the tool name from the selected addon object
            const toolName = Object.keys(selectedAddon)[0];
            const isSelected = selectedAddon[toolName];
            
            if (isSelected) {
              // Find the corresponding tool in originalTools
              const tool = observabilityControl.originalTools.find(t => t.name === toolName);
              
              if (tool && tool.mappinginapi) {
                // Find the addon entry that matches the mappinginapi value
                const addonEntry = this.addons.data.find(entry => 
                  entry.toLowerCase().includes(tool.mappinginapi.toLowerCase())
                );
                if (addonEntry) {
                  managedServices.push({ name: addonEntry });
                  console.log(`Added ${toolName} (${tool.mappinginapi}) as ${addonEntry} to managedServices`);
                }
              }
            }
          });
        }
      }
    }
    //payload for cust
    let payload = {
      "name": "",
      "hypervisor": Hypervisor,
      "purpose": "ipc",
      "vmPurpose": "",
      // flavorId: dedicatedDeploy === "dedicated" ? 3234 : 3261,
      "imageId": imageid,
      "zoneId": zoneid,
      "alertSuppression": true,
      "iops": 1,
      "isKdumpOrPageEnabled": "No",
      ...(tags.length > 0 && { tags }),

      "applicationType": "Container",
      "application": "Containers",
      vmSpecificInput,
      "clusterMode": "High availability",
      "dedicatedDeployment": false,
      "clusterName": name,
      "k8sVersion": k8sversion,
      "circuitId": this.getCopfId(),
      "vApp": "",
      "imageDetails": {
        "valueOSModel": valueosmodel,
        "valueOSMake": valueosmake,
        "valueOSVersion": valueosversion,
        "valueOSServicePack": null
      },
      // "flavorDisk": 50,
      ...(iksBackupDetails && { iksBackupDetails }),
      // ✅ Inline conditional property
      ...(rawPVCs.length > 0 && {
        pvcsEnable: rawPVCs.map(item => ({
          size: item.capacity,
          iops: parseInt(item.classConfiguration?.replace(/\D/g, ''), 10) || null
        }))
      }),
      ...(managedServices.length > 0 && { managedServices }),
      
      ...(cniDriver && {
        networkingDriver: [
          {
            name: cniDriver
          }
        ]
      })
      
    
    }

    
    console.log("final payload", payload)
    console.log("Final Payload in json", JSON.stringify(payload, null, 2));
    this.getLoadingModal("Initiating Instance launch ...");

    this.getLaunchVmData(payload); 
  }
  }
  getLoadingModal(message) {
    this.showSpinner = true;
  }

    getLaunchVmData(payload: any) {
    this.vmService.triggerIKSLaunchVM(
      payload,
      (response) => {
        console.log(response)
        this.showSpinner = false;
        this.showSnackbar({
          message: "Cluster creation request sent successfully",
          subMessage: "Your cluster creation request has been submitted successfully.",
          type: "success"
        });
        this.router.navigate(['/clusters/cluster-list']);
 
      },
      (error) => {
        this.showSpinner = false;
        this.showSnackbar({
          message: "Error in launching the Cluster",
          subMessage: "There was an error while launching your cluster. Please try again.",
          type: "error"
        });
 
      }
    );
  }

  onDisplayAudit(response, message, auditId?: any) {
    const auditDetails = {
      auditId: (response && response.headers && response.headers.get('auditID'))
        ? response.headers.get('auditID') : (response && response.headers && response.headers.get('Auditid') ? response.headers.get('Auditid') : 0),
      message: message
    };

    if (auditId) {
      auditDetails['auditId'] = auditId;
    }
    const dialogRef = this.dialog.open(AuditLogDialogComponent, {
      data: auditDetails
    })
  }

  clusterNameValid: boolean = true;
  clusterNameEntered: string = '';
  clusterListFetched = false;
  clusterList: string[] = [];



  formControlsMap: { [key: string]: FormControl } = {};
  controlSubscriptions: { [key: string]: Subscription } = {};

  onControlReady(event: { name: string, control: FormControl }) {
    const { name, control } = event;
    
    // Optionally save the control reference
    this.formControlsMap[name] = control;

    // Unsubscribe old subscription for this field if exists
    if (this.controlSubscriptions[name]) {
      this.controlSubscriptions[name].unsubscribe();
    }

    // Subscribe to value changes for other fields (not clusterName since it's handled by event)
    if (name !== 'clusterName') {
      this.controlSubscriptions[name] = control.valueChanges.pipe(
        debounceTime(300),
        distinctUntilChanged(),
        map(val => typeof val === 'string' ? val.trim().toLowerCase() : val)
      ).subscribe(value => {
        console.log(`${name} value changed to:`, value);
      });
    }
  }

  get overallExternalValidationPassed(): boolean {
    return this.clusterNameValid; //----->For Each Field validation add a individual flag[boolean].
  }

  tryFetchProjectList(): void {
   // console.log("Inside tryFetchProjectList")
  //  console.log("engId", this.selectedEngId)
    if (this.selectedEngId) {
      this.clusterListFetched = false;

      this.clusterService.getClusterListByEngagement(this.selectedEngId,
        (response: any) => {
          const status = response?.status;
          const innerData = response?.data;

          if (status !== 'error' && Array.isArray(innerData)) {
            this.clusterList = innerData.map(
              (proj: any) => proj?.clusterName?.toLowerCase()
            ) || [];

            console.log("project list:", this.clusterList);

            // Check current entered name immediately after list is available
          //  console.log("checkProjectNameConflict in tryFetchProjectList");
            this.clusterListFetched = true;
            this.checkProjectNameConflict();
            this.cdr.detectChanges();
          } else {
            console.error('Error in Listing Clusters:', innerData);
            this.clusterList = [];
            this.clusterNameValid = false;
            this.clusterListFetched = false;
            this.cdr.detectChanges();
          }
        },
        (error) => {
          console.error("Failed to fetch Cluster list", error);
          this.clusterList = [];
          this.clusterNameValid = false;
          this.clusterListFetched = false;
          this.cdr.detectChanges();
        }
      );
    }
  }

  onClusterNameChanged(clusterName: string): void {
    this.clusterNameEntered = clusterName;
    this.validateClusterName(clusterName);
  }

  onTagNameValidationRequested(event: { tagName: string; defaultEngId: number; index: number }): void {
    const { tagName, index } = event;
    this.validateTagName(tagName, this.selectedEngId, index);
  }

  onCreateTagsRequested(tagsPayload: any[]): void {
    console.log("Creating tags with payload:", JSON.stringify(tagsPayload, null, 2));
    this.createtags(tagsPayload);
  }

  validateTagName(tagName: string, defaultEngId: number, index: number): void {
    this.clusterService.checkTagNameService(tagName, defaultEngId, (response) => {
      console.log("Tag name validation response:", response);
      
      const status = response?.status;
      const data = response?.data;
      
      // Check if tag already exists regardless of status (success or error)
      const isTagAlreadyExists = data?.isTagAlreadyExists;
      
      if (isTagAlreadyExists) {
        // Tag name already exists
        this.createServiceComponent?.onTagNameValidationResult({
          index,
          isValid: false,
          errorMessage: `Tag name "${tagName}" already exists. Please choose a different name.`
        });
        let snackbarData = { 
          message: `Tag name "${tagName}" already exists. Please choose a different name.`, 
          type: 'error' 
        };
        this.showSnackbar(snackbarData);
      } else if (status === 'success') {
        // Tag name is available
        this.createServiceComponent?.onTagNameValidationResult({
          index,
          isValid: true
        });
        let snackbarData = { 
          message: `Tag name "${tagName}" is available.`, 
          type: 'success' 
        };
        this.showSnackbar(snackbarData);
      } else {
        // API error (when status is error but isTagAlreadyExists is not true)
        this.createServiceComponent?.onTagNameValidationResult({
          index,
          isValid: false,
          errorMessage: 'Error checking tag name. Please try again.'
        });
        let snackbarData = { 
          message: `Error checking tag name. Please try again.`, 
          type: 'error' 
        };
        this.showSnackbar(snackbarData);
      }
    }, (error) => {
      console.log("Tag name validation error:", error);
      this.createServiceComponent?.onTagNameValidationResult({
        index,
        isValid: false,
        errorMessage: 'Error checking tag name. Please try again.'
      });
      let snackbarData = { 
        message: `Error checking tag name. Please try again.`, 
        type: 'error' 
      };
      this.showSnackbar(snackbarData);
    });
  }

  validateClusterName(clusterName: string): void {
    // Get the cluster name form control
    const clusterNameControl = this.formControlsMap['clusterName'];
    
    // Check if the form control has any validation errors (pattern, required, etc.)
    if (clusterNameControl && clusterNameControl.errors) {
     // console.log('Cluster name has validation errors:', clusterNameControl.errors);
      this.clusterNameValid = false;
      // Don't call updateClusterNameControlValidation here - let the form validation handle it
      return;
    }

    // Check basic length validation (3-18 characters)
    if (!clusterName || clusterName.trim().length < 3 || clusterName.trim().length > 18) {
    //  console.log('Cluster name length validation failed:', clusterName?.trim().length);
      this.clusterNameValid = false;
      // Don't call updateClusterNameControlValidation here - let the form validation handle it
      return;
    }

    // Only proceed with API call if the field is valid
   // console.log('Cluster name is valid, proceeding with API check');
    
    // Call the new API to check if cluster name exists
    this.clusterService.onCheckClusterName(clusterName,
      (response: any) => {
        const status = response?.status;
        const data = response?.data;

        if (status !== 'error') {
          // If data is empty or has no properties, cluster name doesn't exist (success)
          if (!data || Object.keys(data).length === 0) {
            this.clusterNameValid = true;
            this.updateClusterNameControlValidation(true);
            let snackbarData = { 
              message: `Cluster name "${clusterName}" is available.`, 
              type: 'success' 
            };
            this.showSnackbar(snackbarData);
          } else {
            // If data contains cluster details, cluster name already exists (error)
            this.clusterNameValid = false;
            this.updateClusterNameControlValidation(false);
            let snackbarData = { 
              message: `Cluster name "${clusterName}" already exists. Please choose a different name.`, 
              type: 'error' 
            };
            this.showSnackbar(snackbarData);
          }
        } else {
          console.error('Error in cluster name validation:', data);
          this.clusterNameValid = false;
          this.updateClusterNameControlValidation(false);
          let snackbarData = { 
            message: `Error validating cluster name. Please try again.`, 
            type: 'error' 
          };
          this.showSnackbar(snackbarData);
        }
        this.cdr.detectChanges();
      },
      (error) => {
        console.error("Failed to validate cluster name", error);
        this.clusterNameValid = false;
        this.updateClusterNameControlValidation(false);
        let snackbarData = { 
          message: `Error validating cluster name. Please try again.`, 
          type: 'error' 
        };
        this.showSnackbar(snackbarData);
        this.cdr.detectChanges();
      }
    );
  }

  updateClusterNameControlValidation(isValid: boolean): void {
    const clusterNameControl = this.formControlsMap['clusterName'];
    
    if (clusterNameControl) {
      if (isValid) {
        // Clear any custom validation errors but preserve other validators
        const currentErrors = clusterNameControl.errors;
        if (currentErrors) {
          delete currentErrors['clusterNameExists'];
          if (Object.keys(currentErrors).length === 0) {
            clusterNameControl.setErrors(null);
          } else {
            clusterNameControl.setErrors(currentErrors);
          }
        }
      } else {
        // Set custom validation error that will be detected by the template
        const currentErrors = clusterNameControl.errors || {};
        currentErrors['clusterNameExists'] = true;
        clusterNameControl.setErrors(currentErrors);
        // Also mark as touched so the error shows immediately
        clusterNameControl.markAsTouched();
      }
    }
  }


  checkProjectNameConflict(): void {
    // This method is kept for backward compatibility but the main validation
    // is now handled by validateClusterName method
   // console.log("checkProjectNameConflict called - using new validation approach");
  }

  showSnackbar(data) {
    console.log("Snackbar", data);

    this.snackBar.openFromComponent(SnackbarMessageComponent, {
      data: {
        message: data.message,
        subMessage: data.subMessage,
        type: data.type
      },
      duration: 3000,
      // panelClass: ['custom-snackbar', 'success'],

      panelClass: ['no-default-snackbar'],
      horizontalPosition: 'end',
      verticalPosition: 'top'
    });
  }

  populateDataCenterOptions() {
    if (!this.iksImages || this.iksImages.length === 0) {
      console.warn("No IKS images available for Data Center options");
      return;
    }

    // Create a Map to store unique endpointId, endpointName, and endpoint combinations
    const uniqueEndpoints = new Map<number, { endpointName: string; endpoint: string }>();
    
  //  console.log("Processing IKS images for Data Center options:", this.iksImages.length, "images");
    
    // Extract unique endpointId, endpointName, and endpoint pairs
    this.iksImages.forEach(image => {
      if (image.endpointId && image.endpointName && image.endpoint) {
        uniqueEndpoints.set(image.endpointId, {
          endpointName: image.endpointName,
          endpoint: image.endpoint
        });
    //    console.log(`Found endpoint: ID=${image.endpointId}, Name=${image.endpointName}, Endpoint=${image.endpoint}`);
      }
    });

    // Convert to array format for dropdown options using original endpoint names
    const dataCenterOptions = Array.from(uniqueEndpoints.entries()).map(([endpointId, endpointData]) => {
      return {
        id: endpointData.endpointName,
        itemName: endpointData.endpointName,
        endpointId: endpointId,
        endpointmap: endpointData.endpoint
      };
    });

   // console.log("Data Center options:", dataCenterOptions);

    // Find the Data Center control in stepDefinitions and update its options
    this.stepDefinitions.forEach((step) => {
      step.formControls.forEach((control) => {
        if (control.name === 'datacenter') {
          control.options = dataCenterOptions;
          console.log("Updated Data Center options for control:", control.name);
        }
      });
    });
  }

  populateCNIDriverOptions(cniDrivers: string[]) {
    if (!cniDrivers || cniDrivers.length === 0) {
      console.warn("No CNI drivers available");
      return;
    }

  //  console.log("Processing CNI drivers:", cniDrivers);

    // Convert CNI driver strings to dropdown options format
    const cniDriverOptions = cniDrivers.map(driver => ({
      id: driver,
      itemName: driver
    }));

  //  console.log("CNI Driver options:", cniDriverOptions);

    // Find the CNI Driver control in stepDefinitions and update its options
    this.stepDefinitions.forEach((step) => {
      step.formControls.forEach((control) => {
        if (control.name === 'cnidriver') {
          control.options = cniDriverOptions;
    //      console.log("Updated CNI Driver options for control:", control.name);
        }
      });
    });
  }

  /**
   * Gets the user-friendly display name for node types
   * Maps API values to readable display names
   * @param nodeType - The node type from API (e.g., "generalPurpose")
   * @returns The user-friendly display name (e.g., "General Purpose")
   */
  private getNodeTypeDisplayName(nodeType: string): string {
    if (!nodeType || typeof nodeType !== 'string') return nodeType;
    
    const nodeTypeMapping: { [key: string]: string } = {
      'generalPurpose': 'General Purpose',
      'memoryOptimized': 'Memory Optimized',
      'computeOptimized': 'Compute Optimized'
    };
    
    return nodeTypeMapping[nodeType] || nodeType;
  }

  /**
   * Gets the original flavor category value for API calls
   * @param displayValue - The display value shown in the UI
   * @returns The original flavor category value
   */
  getOriginalFlavorCategory(displayValue: string): string {
    const option = this.nodeTypeOptions.find(opt => opt.itemName === displayValue);
    return option?.originalValue || displayValue;
  }

  /**
   * Gets the currently selected node type's original value for API calls
   * @returns The original node type value
   */
  getCurrentNodeTypeOriginal(): string {
    return this.selectedNodeTypeOriginal || this.selectedNodeType;
  }

  // Tags functionality methods
  onTagKeySelected(event: any, control?: any) {
    this.lastSelectedTag = event.value;

    // Clear search filter and search input when a tag key is selected
    if (control) {
      this.clearSearchFilter(control.name);
      this.clearSearchInput(control.name);
    }

    if (!event.value) {
      this.onTagKeyDeselected(event.value);
    } else {
      // Check if tag already exists to prevent duplicates
      const selected = event.value;
      const tagKey = selected.itemName || selected.name;
      
      if (!this.keyValueTags.find(t => t.key === tagKey)) {
        // Automatically create a key-value pair when a tag is selected
        this.addNewTagRow(control.name);
      }
      
      // Reset selection for next tag
      this.selectedTagKey = null;
    }
  }

  onTagKeyDeselected(value: any) {
    // Handle tag deselection if needed
  }

  addNewTagRow(controlName: string) {
    if (!this.selectedTagKey) return;

    const selected = this.selectedTagKey;
    const newTag = {
      key: selected.itemName || selected.name,
      value: selected.description || '', // Auto-populate with description value
      id: selected.id
    };

    // Add the new tag (duplicate check already done in onTagKeySelected)
    this.keyValueTags.push(newTag);

    // Push to the form
    const stepLabel = this.stepDefinitions[this.currentStepper]?.label;
    // Note: Form handling would need to be implemented based on your form structure
  }

  updateTagValue(tag: any, controlName: string) {
    // Update the form value when tag value changes
    // Note: Form handling would need to be implemented based on your form structure
  }

  removeKeyValueTag(tag: any, controlName: string) {
    this.keyValueTags = this.keyValueTags.filter(t => t.key !== tag.key);

    // Update form
    const stepLabel = this.stepDefinitions[this.currentStepper]?.label;
    // Note: Form handling would need to be implemented based on your form structure
  }

  removeTagFromDropdown(tag: any, controlName: string, event: Event) {
    // Prevent the dropdown from opening when clicking the remove button
    event.stopPropagation();
    
    // Remove the tag from keyValueTags
    this.keyValueTags = this.keyValueTags.filter(t => t.key !== tag.key);

    // Update form
    const stepLabel = this.stepDefinitions[this.currentStepper]?.label;
    // Note: Form handling would need to be implemented based on your form structure
  }

  clearSearchFilter(controlName: string) {
    if (this.filteredOptions[controlName]) {
      delete this.filteredOptions[controlName];
    }
  }

  clearSearchInput(controlName: string) {
    if (this.searchInputValues[controlName]) {
      this.searchInputValues[controlName] = '';
    }
  }

  filterOptions(control: any, event: any) {
    const searchValue = event.target.value.toLowerCase();
    this.searchInputValues[control.name] = searchValue;

    if (!searchValue.trim()) {
      this.clearSearchFilter(control.name);
      return;
    }

    const filtered = control.options.filter((option: any) =>
      option.itemName.toLowerCase().includes(searchValue)
    );

    this.filteredOptions[control.name] = filtered;
  }

  onDropdownClosed(control: any) {
    // Handle dropdown closed event if needed
  }

  shouldShowControl(control: any): boolean {
    // Basic implementation - show all controls by default
    // You can add more complex logic here if needed
    return true;
  }
}
