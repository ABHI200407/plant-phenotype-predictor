import React from 'react'
import {Grid,Paper,TextField, Typography,Button} from '@mui/material';

export const login = () => {
  const heading={fontSize:'2.5rem',fontWeight:'600'};
  const paperStyle={padding:'2rem', margin:"100px auto",borderRadius:"1rem",boxShadow:'10px 10px 10px'};
  const row={display:'flex',margin:'2rem '};
  const btnStyle={margin:'2rem',fontSize:'1.2rem',fontWeight:'700',backgroundColor:"blue",borderRadius:"0.5rem"};
  return (
     <Grid>
        <Paper sx={{ width:{
          xs:'80vw',
          sm:'50vw',
          md:'40vw',
          lg:'30vw',
          xl:'20vw'
        },
        height:'60vh' }}></Paper>
        <Typography style={heading}>Signup</Typography>
        <form>
           <TextField  sx={{label: {fontWeight:'700',fontSize:"1.3rem"}}} style={row} label ="Enter Email" type="email"></TextField>
           <TextField sx={{label: {fontWeight:'700',fontSize:"1.3rem"}}} style={row} label ="Enter Password" type='password'></TextField>
            <Button type="submit"variant="contained" style={btnStyle}>Login</Button>
        </form>
    </Grid>
  )
}
