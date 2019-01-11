function devurl( key ) {

/* The URL of the starlink documents server - used to access on-line
   version of SUN/211. This service determines the current URL of the page
   describing a particular method. */
   var stardocs = "http://www.starlink.ac.uk/cgi-bin/htxserver";

   return stardocs+"/sun211.htx/?"+key;


}
