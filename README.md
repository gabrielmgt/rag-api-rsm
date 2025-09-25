Considerations:

	Changes to the API: Since the example documents are both URLs, I figured it would make more sense to have an url field only so that the backend service doesn't have to interpret the string in the content field. I've left the content field for the sake of completeness but have added a constraint to requests so that either only the url or the content field are used (not both).
	
	I've also created a multi stage docker file even though the container only installs python modules that need no compilation even if it may not be necessary, to learn about multi staging.
	
	