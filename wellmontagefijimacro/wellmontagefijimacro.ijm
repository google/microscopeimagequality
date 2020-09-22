// This script generates whole-plate montages per channel for inspecting data quality.
// Requires that the images already live in a single directory.

/* 
*  Random image patches are sampled (and resampled, if the patch contains no
*  detectable objects). We use the final precentage score as a means to calculate
*  the actual focus score sans the blank patches.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Settings
 */

FOCUS_QUALITY_PLUGIN_PATCH_SIDE_LENGTH = 84; //pixels, do not change this

//image specs

image_width = 875; // pixels
image_height = 512; //pixels

crop_side_length = FOCUS_QUALITY_PLUGIN_PATCH_SIDE_LENGTH*3; // pixels

//channel specs
focus_channel = "CY3-AGP";
channel_list = newArray("DAPI","CY3-AGP");

//Plate specs
rows = newArray("A","B","C","D","E","F","G","H");
columns = newArray("01","02","03","04","05","06","07","08","09","10","11","12");

// filename format
// This macro currently assumes the filenames are of the format: "TestDataSet_WellA01_Site000_DAPI"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* FUNCTIONS */

function append(arr, value) {
  /*
   * Appends value to existing array
   * 
   * Args: arr - array
   *       value - element to add to array
   * 		 
   */
	arr2 = newArray(arr.length+1);
	for (i=0; i<arr.length; i++) {
		arr2[i] = arr[i];
	}
    arr2[arr.length] = value;
    return arr2;
}

function array_contains(arr, element) {
   /*
   * Checks if a given array contains a given element
   * 
   * Args: arr - array
   * 	   element - element
   * 		 
   */
	for (i=0; i<arr.length; i++) {
		if (element == arr[i]) {
			return true;
		} if (i == arr.length-1 && element != arr[i]) {
			return false;
		}
	}
}

function index(arr, element) {
	/*
	 * Finds the index of an element in an array
	 * 
	 * Args: 
	 */
	for (i=0; i<arr.length; i++) {
		if (arr[i]==element) {
			return i;
		}
	}
}

function make_dir(path_name, make_new) {
	/*
	 * Checks to see if directory exists and creates new one if doesn't
	 * 
	 * Args: path_name - directory name and path
	 * 
	 * Returns: New directory if does not exist, otherwise prints "Directory already exists!"
	 */
	new_path_base = substring(path_name, 0, lengthOf(path_name)-1);
	new_path = new_path_base + "/";
	counter = 0;
	if (make_new == false) {
		if (!File.isDirectory(new_path)) {
			File.makeDirectory(new_path);
		} else {
			print("Directory already exists!");
		}
	} else {
		while (File.isDirectory(new_path)) {
			counter++;
			new_path = new_path_base + "-" + counter + "/";
		}
		File.makeDirectory(new_path);
		return new_path;
	}
}

function randint(start, end) {
	// Uniform random, inclusive.
	//a = randint(1,3);
  a=random();
  return round(a*(end-start) + start);
}


function make_random_rectangle(imageID) {
	/*
	 * Randonly selects a rectangle ROI of an image
	 * 
	 * Args: imageID = image to select ROI from
	 */
	counter = 0;
	selectImage(imageID);
	makeRectangle(randint(0, image_width - crop_side_length), randint(0,image_height - crop_side_length), crop_side_length, crop_side_length); //random crop
	getRawStatistics(nPixels, mean, min, max, std, histogram);
	// criteria of min/max is to only select ROIs that aren't blank
	while(min/max > 0.1) {
		if (counter == random_iterations) {
			// max number of iterations
			break;
		} else {
			counter++;
			makeRectangle(randint(0, image_width - crop_side_length), randint(0,image_height - crop_side_length), crop_side_length, crop_side_length); //random crop
			getRawStatistics(nPixels, mean, min, max, std, histogram);
		}
	}
	run("Crop");
	run("Enhance Contrast", "saturated=0.35");
}

function random_tile_crop(well, channel, savepath, fileName, start, end, n) {
	/*
	 * Creates random ROI per well and measures intensity — if intensity is less than 90%, reselect a different random ROI
	 * 
	 * Args: well - well of interest
	 * 		 channel - channel/OC of interest
	 * 		 savepath - path of file to be saved
	 * 		 fileName - data directory (ex. E:/20190717_194041_709/)
	 * 		 
	 * Returns: randomly chosen ROI that meets standards and creates montage of all wells
	 */

	//Randomly select tile from available tiles of a single well
	//Once tile is selected, randomly select ROI
	if (n == 1) {
		tile_str = "000";
		run("Image Sequence...", "open="+fileName+"FOO.tif file=(.*"+well+"_Site"+tile_str+"_"+channel+") sort");
		inFile = getImageID();
		make_random_rectangle(inFile);
	} else {
		for (i=0; i < num_random_tiles; i++) {
			a = randint(start+i*(end-start)/n, start+(i+1)*(end-start)/n);
			while (lengthOf(""+a) < 3) {
				a = "0" + a;
			}
			run("Image Sequence...", "open="+fileName+"FOO.tif file=(.*"+well+"_Site"+a+"_"+channel+") sort");
			inFile = getImageID();
			make_random_rectangle(inFile);
		}
	}

	// Create montage of all selected ROIs from a single well and save as tiff
	if (n < 4){
		image_ID = inFile;
	} else {
		run("Images to Stack");
		inFile = getImageID();
		run("Make Montage...", "columns="+tile_side_length+" rows="+tile_side_length+" scale=1");
		image_ID = getImageID();
		selectImage(inFile);
		close();
	}
	selectImage(image_ID);
	saveAs("Tiff",savepath);
	close();
}
function get_and_save_image_intensity(crop_side_length, num_random_tiles, total_rows, save_tiff, save_jpg) {
	// Add mean intensity to image and save as jpg.
    getRawStatistics(nPixels, mean, min, max, std, histogram);
	//saveAs("Tiff",save_tiff);
	text = "Mean Intensity: "+ mean + " (" + std + " SD)";
	setFont("SansSerif", 75, " antialiased");
	montage_height = crop_side_length*num_random_tiles*total_rows;
	if (num_random_tiles > 1) {
		montage_height = montage_height*0.5; // divide by 2 since each well is made of a 2x2 montage and only 2 ROIs contribute to height
	}
	makeText(text, 45, montage_height - (montage_height*0.15));
	run("Add Selection...", "stroke=black fill=#FFFFFF new");
	run("Enhance Contrast", "saturated=0.35");
	saveAs("Jpeg", save_jpg);
	close();
}
function quality_analysis(path_name, save_name, montageType, min_threshold, max_threshold) {
	/*
	 * Conducts quality analysis
	 * 
	 * Args: minimum and maximum values for thresholding
	 * 
	 * Returns: percentage score of focus quality for plate
	*/
	run("Image Sequence...", "open="+path_name+"FOO.tif file=(.*Well_"+montageType+".*"+focus_channel+".tif*) sort");
	inFile = getImageID();

	run("Make Montage...", "columns="+total_cols+" rows="+total_rows+" scale=1");
	montage = getImageID();
	selectImage(inFile);
	close();
	selectImage(montage);
	saveAs("Tiff",save_name+".tif");
	if (intensity_montages[index(channel_list,focus_channel)] == true) {
		get_and_save_image_intensity(crop_side_length, num_random_tiles, total_rows, save_name+".tif", save_name+".jpg");
	}
	open(save_name+".tif");
	File.delete(save_name+".tif");
	run("Enhance Contrast", "saturated=0.35");
	print("Computing focus quality...");
	setBatchMode(false);
	tilecounty = getHeight()/FOCUS_QUALITY_PLUGIN_PATCH_SIDE_LENGTH;
	tilecountx = getWidth()/FOCUS_QUALITY_PLUGIN_PATCH_SIDE_LENGTH;
	run("Microscope Image Focus Quality", "originalImage="+save_name+".tif tilecountx="+tilecountx+" tilecounty="+tilecounty+" createprobabilityimage=true overlaypatches=true solidpatches=false borderwidth=4");
	selectWindow(montage_fileName + ".tif");
	saveAs("Jpeg", save_name+"-focus.jpg");
	close();
	// The probabilities is a stack with 11 slices, corresponding to probability of 1, 4, ..., 31 pixel blur.
	// We sum the probabilities corresponding to 1, 4 and 7 pixel blurs here, as the acceptable focus threshold.
	selectWindow("Probabilities");
	run("Make Substack...", "channels=1-3");
	run("Z Project...", "projection=[Sum Slices]");
	selectWindow("SUM_Probabilities-1");
	setAutoThreshold("Default dark");
	setThreshold(min_threshold, max_threshold);
	call("ij.plugin.frame.ThresholdAdjuster.setMode", "B&W");
	setOption("BlackBackground", true);
	run("Convert to Mask");
	getRawStatistics(nPixels, mean, min, max, std, histogram);

	focus_score = round(100*mean/255);
	print("Percentage patches in-focus: " + focus_score +"%");
	close();
	selectWindow("Probabilities");
	close();
	selectWindow("Probabilities-1");
	close();
	
	open(save_name+"-focus.jpg");
	text = " Percentage Score = "+ focus_score + "%  ";
	setFont("SansSerif", 75, " antialiased");
	montage_height = crop_side_length*num_random_tiles*total_rows;
	if (num_random_tiles > 1) {
		montage_height = montage_height*0.5; // divide by 2 since each well is made of a 2x2 montage and only 2 ROIs contribute to height
	}
	makeText(text, 45, montage_height - (montage_height*0.11));
	run("Add Selection...", "stroke=black fill=#FFFFFF new");
	run("Select None");
	saveAs("Jpeg", save_name+"-focus.jpg");
	close();
	setBatchMode(true);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


channel_string = "";
for (ch=0;ch<channel_list.length;ch++){
	if (ch == channel_list.length-1) {
		channel_string = channel_string+channel_list[ch];
	} else {
		channel_string = channel_string+channel_list[ch]+", ";
	}
}

total_rows = rows.length;
total_cols = columns.length;

getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, msec);
dd_month = month + 1;
dd_day = dayOfMonth;
if (dd_month < 10) {
	dd_month = "0" + dd_month;
}


Dialog.create("Plugin Assumptions & Requirements");
Dialog.addMessage("Requirements:");
Dialog.addMessage("- Need to have Microscope Focus Quality Classifier plugin enabled; all images need to be in a single folder.");
Dialog.addMessage("Settings (change these by editing constants at the top of the macro):");
Dialog.addMessage("- Image height = "+image_height+", image width = "+image_width);
Dialog.addMessage("- Total rows = "+total_rows+", total columns = "+total_cols+", total # of wells = "+ total_cols*total_rows);
Dialog.addMessage("- Filename format: TestDataSet_WellA01_Site000_DAPI");
Dialog.addMessage("- Channels = "+channel_string);
Dialog.addMessage("- Focus channel = "+focus_channel);
Dialog.show();
 
Dialog.create("Focus Quality Check");
Dialog.addString("Input Image Directory: ", "Example: C:/data_directory/",30);
Dialog.addString("Output subfolder: ", year + dd_month + dd_day);
Dialog.addNumber("Max number of tiles per well to analyze: ", 1);
Dialog.addCheckbox("Check box if you want to skip intensity analysis and only analyze focus", false);
Dialog.addCheckbox("Check box if there are empty wells", false);
Dialog.show();

// Get information from GUI and assign as constants
data_directory = Dialog.getString();
analysis_date = Dialog.getString();
num_of_tiles = Dialog.getNumber();
focus_only = Dialog.getCheckbox();
empty_wells = Dialog.getCheckbox();

if(!data_directory.endsWith('/')) {
	exit("Input Image Directory must end with '/'");
}


//Array of true or falses. Determines what channels to analyze intenisty
intensity_montages = newArray(channel_list.length);
if (focus_only == false) {
	Dialog.create("Channel Chooser for Intensity Montage")
	for (ch = 0; ch<channel_list.length; ch++) {
		Dialog.addCheckbox(channel_list[ch], true);
		if (ch != channel_list.length-1) {
			Dialog.addToSameRow();
		}
	}
	Dialog.show();

	for (ch = 0; ch<channel_list.length; ch++) {
		intensity_montages[ch] = Dialog.getCheckbox();
	}
} 
else {
	for (ch = 0; ch<channel_list.length; ch++) {
		intensity_montages[ch] = false;
	}
}

montageFolderName = analysis_date + "_Montages";
montage_fileName = focus_channel + "-Montage";

inNames = newArray(data_directory);
random_iterations = 200;
recompute_all=false;

//Select wells to analyze
wells = newArray;

for (i=0; i<rows.length; i++) {
	for (j=0; j<columns.length; j++) {
		well = rows[i]+columns[j];
		wells = append(wells, well);
	}
}

default_list = newArray(96);
for (k=0; k<default_list.length; k++) {
	default_list[k] = true;
}

skip_wells = newArray;

if (empty_wells == true) {
	Dialog.create("Well Chooser");
	Dialog.addCheckboxGroup(8,12, wells, default_list);
	Dialog.show();
	for (i=0; i<default_list.length; i++) {
		default_list[i] = Dialog.getCheckbox();
		if (default_list[i] == false) {
			skip_wells = append(skip_wells, wells[i]);
		}
	}
}

all_channels = newArray(focus_channel);
for (ch = 0; ch<channel_list.length; ch++) {
	if (intensity_montages[ch] == true){
		if (index(channel_list,focus_channel) != ch) {
			all_channels = append(all_channels, channel_list[ch]);
		}
	}
}
total_channels = all_channels.length;


min_thresh = 0.5000;
max_thresh = 1000000000000000000000000000000.0000;

if (num_of_tiles < 4) {
	num_random_tiles = 1;
	tile_side_length = 1;
} else {
	num_random_tiles = 4;
	tile_side_length = 2;
}

random_seed = second;
start_tile = 0;
end_tile = num_of_tiles - 1;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


run("Close All");
for(scan=0; scan<inNames.length; scan++) {
	inName = inNames[scan];
	print(inName);
	pathName = make_dir(inName + montageFolderName + "/", true);
	random("seed",random_seed);
	files_to_delete = newArray();
	
	setBatchMode(true); //so we don't flash all the images 

	// Generate temporary montages of each well.
	print("starting Zoom montage creation...");
	for(c=0;c<total_channels;++c) {
		ch = all_channels[c];
		print("Processing Channel: "+ch);
		for (x=0;x<rows.length;++x) {
			//print("Processing row: " +rows[x] );
			for(y=0;y<columns.length;++y) {
				well = rows[x] + columns[y];
				if (skip_wells.length == 0) {
					skip = false;
				} else {
					skip = array_contains(skip_wells, well);
				}
				savename = ""+pathName+"Well_zoom_"+well+"_"+ch+".tif";
				files_to_delete = append(files_to_delete, savename);
				if (skip == true) {
					newImage("blank", "16-bit black", 504, 504, 1);
					image_ID = getImageID();
					selectImage(image_ID);
					saveAs("Tiff",savename);
					close();
				} else {
					if( !File.exists( savename ) | recompute_all ) {
						random_tile_crop(well, ch, savename, inName, start_tile, end_tile, num_of_tiles);
					}
				}
			}
		}
		
		print("Building Plate montage...");
		montage_type = "zoom";
		savename = ""+pathName+montage_fileName; 
		//savename_jpg = ""+pathName+montage_fileName+".jpg";
		if(ch == focus_channel) {
			quality_analysis(pathName, savename, montage_type, min_thresh, max_thresh);
		} else {
			setBatchMode(true);
			save_tiff = ""+pathName+ch+"-Montage.tif"; 
			save_jpg = ""+pathName+ch+"-Montage.jpg";
			run("Image Sequence...", "open="+pathName+"FOO.tif file=(.*Well_"+montage_type+".*"+ch+".tif*) sort");
			inFile = getImageID();
			// Unfortunately having a border prevents auto-contrast from working.
			run("Make Montage...", "columns="+total_cols+" rows="+total_rows+" scale=1");
			montage = getImageID();
			selectImage(inFile);
			close();
			selectImage(montage);
			get_and_save_image_intensity(crop_side_length, num_random_tiles, total_rows, save_tiff, save_jpg);
		}
	}
	// Delete temporary files.
	for(i=0; i<files_to_delete.length; i++) {
		File.delete(files_to_delete[i])
	}
}	

exit("Done running! Results are at: " + pathName);
print("Done");
