import React from 'react';
import { Link } from 'react-router-dom';
import {AppBar, Typography, Toolbar, Button} from '@mui/material';
import { Logout } from './logout';

export const navbar = () => {
  return (
    <> 
       <AppBar sx={{bgcolor:'lightblue'}}>
          <Toolbar>
              <Typography variant="h4" sx={{flexGrow:1}}>Pheno-Predict</Typography>
              <Button style={button} color= "error" variant="contained" to="/login" component={Link}>Login</Button>
              <Button style={button} color="success" variant="contained" to="/signup" component={Link}>Signup</Button>
              <Logout/>
          </Toolbar>
        </AppBar>


    </>
    //<div>navbar</div>



  )
}


